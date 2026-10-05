from collections import defaultdict, deque
from typing import NamedTuple
import time
import timeout_decorator
from agent_baselines import Agent

# Order 'kinds', as appearing in the order string (movement has kind '-')
MOVE, RETREAT, SUPPORT, CONVOY, HOLD, BUILD, DISBAND = '-', 'R', 'S', 'C', 'H', 'B', 'D'

# Type alias for plans... (own unit) province : the order string which that unit will be given
Plan = dict[str, str]

# Assigned distance for provinces with no route to a target centre, + distance cap on otherwise dominating stranded units
MAX_DISTANCE = 10



def province_of(location: str) -> str:
    '''
    Returns province names stripped of the coast: 'STP/SC' -> 'STP'.
    Coasts are only relevant to the engine for order legality.
    Everywhere in this code, a province refers to only the province itself.
    map_graph, all_centres, even engine orderable locations are all keyed without coasts.
    Used outside of StudentAgent too, so kept at top.
    '''
    return location.split('/')[0]


def province_of_unit(unit: str) -> str:
    '''
    Returns the province that a unit string stands on: 'F STP/SC' -> 'STP', '*A MUN' (dislodged) -> 'MUN'.
    '''
    return province_of(unit.split()[1])


def type_of_unit(unit: str) -> str:
    '''
    Returns the type of a unit string, 'A' army or 'F' fleet: 'F STP/SC' -> 'F', '*A MUN' (dislodged) -> 'A'.
    '''
    return unit.split()[0].lstrip('*')


class ParsedOrder(NamedTuple):
    '''
    An order string split into the relevant parts...
        unit_type: 'A' army or 'F' fleet - the unit being ordered.
        location: province of the unit being ordered, coast stripped.
        kind: '-' move, 'R' retreat, 'S' support, 'C' convoy, 'H' hold, 'B' build, 'D' disband.
        destination: province of the target in the order (move/supported), coast stripped.
        supported_location: province of the unit being supported/convoyed if at all, else None.
    '''
    unit_type         : str
    location          : str
    kind              : str
    destination       : str | None
    supported_location: str | None


def parse_order(order: str) -> ParsedOrder | None:
    '''
    'F NTH S A EDI - YOR' -> ParsedOrder('F', 'NTH', 'S', 'YOR', 'EDI') | None for 'WAIVE'.
    '''
    parts = order.split()
    if len(parts) < 3: # Waive
        return None

    unit_type, location, kind = parts[0], province_of(parts[1]), parts[2]

    # Move or Retreat:
    #   'F IRI - MAO (VIA ... if army)' (move),
    #   'A WAL R LON' (retreat)
    if kind in (MOVE, RETREAT):
        return ParsedOrder(unit_type, location, kind, province_of(parts[3]), None)

    # Support or Convoy:
    #   'A WAL S F LON' (hold support),
    #   'F NTH S A EDI - YOR' (move support),
    #   'F NWG C A NWY - EDI' (convoy)
    if kind in (SUPPORT, CONVOY):
        destination = province_of(parts[6]) if len(parts) >= 7 else None
        return ParsedOrder(unit_type, location, kind, destination, province_of(parts[4]))

    # Hold, Build, or Disband:
    #   'A LON H' (hold),
    #   'A LON B' (build),
    #   'A LON D' (disband)
    return ParsedOrder(unit_type, location, kind, None, None)


class PredictedOutcome(NamedTuple):
    '''
    What we predict a plan achieves once the turn resolves...
        final_units: (predicted end) province : type of our unit there - 'A' for army or 'F' for fleet.
        dislodged_locations: {provinces where our attack is predicted to push an enemy unit off}.
        num_failed_attacks: our attacks predicted to bounce, one wasted unit-turn each.
        num_wasted_supports: our supports for a move or hold the plan does not actually make.
        hold_support_at: (own unit) province : count of our supports for the unit there holding.
    '''
    final_units        : dict[str, str]
    dislodged_locations: set[str]
    num_failed_attacks : int
    num_wasted_supports: int
    hold_support_at    : dict[str, int]



class StudentAgent(Agent):
    '''
    Beam search agent: builds a plan (keyed set of orders) one unit at a time,
    keeping the best beam_width partial plans at each step via the heuristic in _score_plan.
    '''
    @timeout_decorator.timeout(1)
    def __init__(self, agent_name: str='v4-51.8%'):
        super().__init__(agent_name)

        self.debug = False

        self.time_budget = 0.6 # Set well under one-second for safety

        # Agent parameters
        self.beam_width = 16 # = 1 makes this greedy search (possible 'improvement' greedy -> beam)
        self.candidate_size = 16 # legal orders considered per unit

        # Score weights, term name : multiplier in _evaluate_outcome. Tuned by experiment, see results/.
        self.score_weights: dict[str, float] = {
            'capture_centre': 30.0,   # end a Fall turn on a centre we do not own - ownership only changes then
            'occupy_centre' :  8.0,   # end a Spring turn on one - no capture yet, but in place for the Fall
            'keep_centre'   :  2.0,   # end the turn on an own centre
            'progress'      :  3.0,   # per hop closer to the nearest target centre, along that unit type's routes
            'dislodge_unit' :  4.0,   # predicted to push an enemy unit off the province it stands on
            'failed_attack' : -2.0,   # predicted bounce: a wasted unit-turn
            'wasted_support': -2.0,   # support for a move or hold the plan does not actually make
            'expose_centre' : -6.0,   # an own centre left empty, times the chance it is attacked (enemy_attack_chance_at)
            'hold_support'  :  2.0,   # a hold support for an own unit sitting on an own centre, times the chance it is attacked (enemy_attack_chance_at)
            'pressure'      :  6.0    # per attacker next to an enemy-occupied target centre, up to attackers_needed_at
        }

        # Prune priorities, what the order does : priority. Higher is kept first.
        self.prune_priorities: dict[str, float] = {
            'empty_centre'       : 10.0,   # move into an empty centre, whoever owns it
            'enemy_unit_centre'  :  8.0,   # move into a centre an enemy unit stands on
            'empty_province'     :  3.0,   # move into an empty non-centre province
            'own_unit_province'  : -5.0,   # move into a province one of our own units stands on
            'enemy_unit_province':  4.0,   # move into a non-centre province an enemy unit stands on
            'support'            :  5.0,   # any support, for a move or a hold
            'hold'               :  1.0    # hold
        }

        # Retreat priorities, feature of the destination : priority. Highest is taken.
        self.retreat_priorities: dict[str, float] = {
            'home_centre'        : 10.0,   # retreat onto one of our home centres
            'other_centre'       :  8.0,   # retreat onto any other centre
            'other_province'     :  2.0,   # retreat onto a non-centre province
            'enemy_neighbour'    : -1.5,   # per enemy unit next to the destination
            'next_to_home_centre':  1.0    # destination (not home centre) borders one of our home centres
        }

        # Opponent model
        # Rate priors, rate name : Bayesian starting guess for the learned rate (see _observe_opponents)
        self.rate_priors: dict[str, float] = {
            'defence': 0.5,    # a unit that could support one of its own units' hold does - even guess until shown otherwise
            'attack' : 1.0     # a unit that could move into our ground does - assumed until it shows otherwise
        }
        self.num_prior_observations = 3 # pretend observations behind each rate prior, so one early turn does not swing a rate
        self.opponent_counts: dict[str, dict[str, int]] = {} # power : {count name : count}, over every movement phase so far

        self.is_fall = False # True in a Fall phase (movement or retreat) - the only time when centre ownership changes
        self.occupied_by: dict[str, str] = {} # (occupied) province : power whose unit stands there
        self.unit_type_at: dict[str, str] = {} # (occupied) province : type of unit there, 'A' or 'F'
        self.centre_owned_by: dict[str, str] = {} # centre : power owning it, absent if unowned
        self.army_distance_to_target: dict[str, int] = {} # (army-reachable) province : hops to nearest target centre along army routes only
        self.fleet_distance_to_target: dict[str, int] = {} # (fleet-reachable) province : hops to nearest target centre along fleet routes only

        # Attacking their centres
        self.enemy_defence_strength_at: dict[str, float] = {} # (all) province : defence strength of the enemy unit there, 0 if empty or ours
        self.attackers_needed_at: dict[str, int] = {} # (enemy-occupied target) centre : own units needed next to it to dislodge the enemy unit
        self.army_pressure_centres: dict[str, list[str]] = {} # (army) province : [enemy-occupied target centres that an army there could move into]
        self.fleet_pressure_centres: dict[str, list[str]] = {} # (fleet) province : [enemy-occupied target centres that a fleet there could move into]

        # Defending our centres (will change to mirror attacking)
        self.enemy_attack_chance_at: dict[str, float] = {} # own centre : chance an enemy unit that could move in attacks this turn, 0 if none


    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        '''
        Start of a game: keeps our own copy of the board and builds the map graphs once,
        since they are constant and are too slow to rebuild per turn.
        '''
        self.game = game
        self.power_name = power_name
        self.all_centres = set(self.game.map.scs) # {every centre on the board}
        self.home_centres = set(self.game.map.homes.get(power_name, [])) # centres we started with (fixed) - only centres we can build at

        # Map graphs (adjacency lists), one per unit type
        self.army_graph = self._build_map_graph('A') # province : {provinces an army can move to}
        self.fleet_graph = self._build_map_graph('F') # province : {provinces a fleet can move to}
        self.map_graph = {location: self.army_graph.get(location, set()) | self.fleet_graph.get(location, set())
                          for location in set(self.army_graph) | set(self.fleet_graph)} # province : {provinces either type can move to}

        # Opponent model counts, per rate: how often a unit could have done it (opportunities) and how many times it did it
        self.opponent_counts = {
            power: {
                'defence_opportunities': 0,
                'defences'             : 0,
                'attack_opportunities' : 0,
                'attacks'              : 0
            }
            for power in sorted(self.game.powers)
            if power != self.power_name
        }


    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        '''
        End of a phase: replays every power's orders onto our own board copy,
        to keep in step with the real game.
        '''
        # Before processing, so the board is the one the orders were given on (otherwise would have to roll board back)
        if self.game.phase_type == 'M':
            self._observe_opponents(all_power_orders)

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()


    @timeout_decorator.timeout(1)
    def get_actions(self) -> list[str]:
        '''
        Returns the list of orders (call it an 'unkeyed' Plan) for the current game state.
        '''
        try:
            if self.game.phase_type == 'M':
                return self._movement_orders()
            if self.game.phase_type == 'R':
                return self._retreat_orders()
            return self._adjustment_orders()
        except Exception:
            if self.debug:
                raise
            return []



##### Phases #####

    def _movement_orders(self) -> list[str]:
        '''
        Spring/Fall movement phase: calling beam search.
        '''
        deadline = time.perf_counter() + self.time_budget

        legal_orders = self.game.get_all_possible_orders() # province : [legal order strings]
        moveable_locations = self.game.get_orderable_locations(self.power_name)

        if not moveable_locations: # Early exit
            return []

        self._read_board()

        # The following is built once per turn here rather than hundreds of times per turn in _score_plan
        
        # Attacking their centres: strength each enemy unit could defend with, from learned defence rates
        self.enemy_defence_strength_at = {location: self._enemy_defence_strength(location) for location in self.map_graph}
        # Fewest attackers that beat it (a tie bounces the attacker): one moving unit + supporters
        self.attackers_needed_at = {centre: int(self.enemy_defence_strength_at[centre]) + 1 for centre in self.target_centres if self.enemy_defence_strength_at.get(centre, 0.0) > 0}

        # Defending our centres : chance each is attacked, from learned attack rates
        self.enemy_attack_chance_at = {centre: self._enemy_attack_chance(centre) for centre in self.own_centres}

        # Each unit type measured only along the routes it can take, like the distance maps
        self.army_pressure_centres = self._build_pressure_centres(self.army_graph)
        self.fleet_pressure_centres = self._build_pressure_centres(self.fleet_graph)

        plan = self._beam_search(moveable_locations, legal_orders, deadline)

        return list(plan.values())


    def _adjustment_orders(self) -> list[str]:
        '''
        Winter adjustments phase: greedily build the units that start closest to a target centre (optimal by default on this metric).
        Importantly, this is based on their own unit type's routes - so an island power builds fleets, and landlocked centres build armies.
        For example: England army units are considered far => won't build armies there.

        Letting engine disband for now if over-strength...
        TODO: disband the army/fleet least useful (furthest from fighting maybe?).
        '''
        legal_orders = self.game.get_all_possible_orders()
        adjustable_locations = self.game.get_orderable_locations(self.power_name)
        num_builds_allowed = len(self.game.get_centers(self.power_name)) - len(self.game.get_units(self.power_name))

        if not adjustable_locations: # Early exit
            return []

        self._read_board()

        # Score every legal build
        scored_builds: list[tuple[int, str, str]] = [] # (hops to nearest target centre, build order, home centre)
        scored_disbands = [] # see TODO
        for location in sorted(adjustable_locations):
            for order in legal_orders.get(location, []):
                parsed_order = parse_order(order)
                if parsed_order is not None and parsed_order.kind == BUILD:
                    distance = self._distance_to_target(parsed_order.location, parsed_order.unit_type)
                    scored_builds.append((distance, order, parsed_order.location))
                elif parsed_order is not None and parsed_order.kind == DISBAND:
                    pass # see TODO
        scored_builds.sort() # Scored and sorted by closest to a target centre, ties broken by order (alphabetical) for determinism

        orders = []
        built_locations = set() # One unit built per home centre
        for _, order, location in scored_builds:
            if len(orders) < num_builds_allowed and location not in built_locations:
                orders.append(order)
                built_locations.add(location)

        return orders


    def _retreat_orders(self) -> list[str]:
        '''
        Retreat phase: retreat to the safest and most strategic location.
        Priorities:
            1. A home centre.
            2. Any other centre (neutral or enemy).
            3. A province that is not a centre (bonus for next to home centre).
        Retreat destinations are always empty (as offered by engine).
        '''
        legal_orders = self.game.get_all_possible_orders()
        retreatable_locations = self.game.get_orderable_locations(self.power_name)

        if not retreatable_locations: # Early exit
            return []

        self._read_board()

        orders = []
        for location in sorted(retreatable_locations):
            retreat_options = []
            for order in legal_orders.get(location, []):
                parsed_order = parse_order(order)
                if parsed_order is not None and parsed_order.kind == RETREAT:
                    retreat_options.append(order)

            if not retreat_options:
                continue

            best_order, best_score = None, -float('inf')
            for order in sorted(retreat_options): # Sorted so ties break the same way every run
                destination = parse_order(order).destination
                score = 0.0

                if destination in self.home_centres:
                    score += self.retreat_priorities['home_centre']
                elif destination in self.all_centres:
                    score += self.retreat_priorities['other_centre']
                else:
                    score += self.retreat_priorities['other_province']

                num_enemy_neighbours = 0
                for neighbour in self.map_graph.get(destination, ()):
                    occupant = self.occupied_by.get(neighbour)
                    if occupant is not None and occupant != self.power_name and self._can_reach(neighbour, destination):
                        num_enemy_neighbours += 1
                score += num_enemy_neighbours * self.retreat_priorities['enemy_neighbour']

                if destination not in self.home_centres and self.map_graph.get(destination, set()) & self.home_centres:
                    score += self.retreat_priorities['next_to_home_centre']

                if score > best_score:
                    best_order, best_score = order, score

            if best_order is not None:
                orders.append(best_order)

        return orders



##### Board and Map #####

    def _read_board(self):
        '''
        Initialises relevant data structures, reads the board at the start of every phase (movement, retreat, adjustment) - everything shared between phases:
            - who stands where (occupied_by, unit_type_at),
            - who owns which centre (own_centres, centre_owned_by),
            - the season (is_fall),
            - which centres we are after (target_centres) and how far each unit type is from one.
        '''
        self.occupied_by = self._build_occupied_by()
        self.unit_type_at = self._build_unit_type_at()
        self.centre_owned_by = self._build_centre_owned_by()
        self.own_centres = set(self.game.get_centers(self.power_name))
        self.is_fall = self.game.phase.split()[0] == 'FALL' # 'FALL 1901 MOVEMENT' for example

        # Target centres refers to every centre that is not ours
        self.target_centres = [centre for centre in sorted(self.all_centres) if self.centre_owned_by.get(centre) != self.power_name]

        # Each unit type measured only along the routes it has accessible
        self.army_distance_to_target = self._build_distance_to(self.target_centres, self.army_graph)
        self.fleet_distance_to_target = self._build_distance_to(self.target_centres, self.fleet_graph)


    def _distance_to_target(self, location: str, unit_type: str) -> int:
        '''
        Returns hops from the province to the nearest target centre specifically for that unit type ('A' or 'F'),
        or MAX_DISTANCE if it has no route.
        '''
        if unit_type == 'A':
            return self.army_distance_to_target.get(location, MAX_DISTANCE)
        return self.fleet_distance_to_target.get(location, MAX_DISTANCE)


    def _graph_of(self, unit_type: str) -> dict[str, set[str]]:
        '''
        Returns the map graph for that unit type: army_graph for 'A', fleet_graph for 'F'.
        '''
        if unit_type == 'A':
            return self.army_graph
        return self.fleet_graph


    def _can_reach(self, location: str, destination: str) -> bool:
        '''
        Returns True if the unit standing on location could move into destination in a single move, along its own unit type's routes.
        This is what a move, support or attack needs: abuts, precomputed per province by _build_map_graph.
        Slightly generous for a fleet on a split-coast province compared to abuts (counts as reaching either coast's neighbours).
        '''
        unit_type = self.unit_type_at.get(location)
        return unit_type is not None and destination in self._graph_of(unit_type).get(location, ())


    def _pressure_centres(self, location: str, unit_type: str) -> list[str]:
        '''
        Returns the enemy-occupied target centres that a unit of this type ('A' or 'F') on this province could move into,
        or [] if none.
        '''
        if unit_type == 'A':
            return self.army_pressure_centres.get(location, [])
        return self.fleet_pressure_centres.get(location, [])


    def _build_centre_owned_by(self) -> dict[str, str]:
        '''
        Returns (owned) centre : power owning it.
        '''
        return {centre: power for power in self.game.powers for centre in self.game.get_centers(power)}


    def _build_occupied_by(self) -> dict[str, str]:
        '''
        Returns (occupied) province : power whose unit stands there.
        '''
        occupied_by = {}
        for power in self.game.powers:
            for unit in self.game.get_units(power):
                occupied_by[province_of_unit(unit)] = power

        return occupied_by


    def _build_unit_type_at(self) -> dict[str, str]:
        '''
        Returns (occupied) province : type of unit there 'A' or 'F'.
        Needed by _can_reach: occupied_by only records the power, not whether the unit is an army or a fleet.
        '''
        unit_type_at = {}
        for power in self.game.powers:
            for unit in self.game.get_units(power):
                unit_type_at[province_of_unit(unit)] = type_of_unit(unit)

        return unit_type_at


    def _build_map_graph(self, unit_type: str) -> dict[str, set[str]]:
        '''
        Returns province : {adjacent provinces} for one unit type, 'A' army or 'F' fleet.
        Coasts fold into their province AFTER adjacency is checked ('SPA/NC' becomes 'SPA').
        '''
        locations = sorted({location.upper() for location in self.game.map.locs})

        graph = defaultdict(set)
        for source in locations:
            for target in locations:
                if province_of(source) == province_of(target):
                    continue
                if self.game.map.abuts(unit_type, source, '-', target):
                    graph[province_of(source)].add(province_of(target))

        return graph


    def _build_pressure_centres(self, graph: dict[str, set[str]]) -> dict[str, list[str]]:
        '''
        Returns (province next to target centre) : [enemy-occupied target centres reachable in one move along graph's edges].
        This is what attracts units to line up attacks on enemy-occupied target centres where needed.
        '''
        pressure_centres = {}
        for location, neighbours in sorted(graph.items()):
            centres = [neighbour for neighbour in sorted(neighbours) if neighbour in self.attackers_needed_at]
            if centres:
                pressure_centres[location] = centres

        return pressure_centres


    def _build_distance_to(self, sources: list[str], graph: dict[str, set[str]]) -> dict[str, int]:
        '''
        Returns (reachable) province : hops to the nearest source, moving only along the graph's edges.
        This is what gives a unit reason to move when no adjacent centres (weights moves towards smaller hop number).
        Runs BFS from all sources at once.
        '''
        distances = {source: 0 for source in sources if source in graph} # Inland centres not in fleet graph (cannot be reached)
        fringe = deque(source for source in sources if source in graph)

        while fringe:
            current = fringe.popleft()
            for neighbour in sorted(graph.get(current, ())):
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1
                    fringe.append(neighbour)

        return distances


    def _feature(self, name: str, default):
        '''
        Reads an ablation flag - flags are read when they are used rather than stored in the constructor.
        Test harness attaches features after __init__.
        '''
        return getattr(self, 'features', {}).get(name, default)



##### Search #####

    def _beam_search(self, moveable_locations: list[str], legal_orders: dict[str, list[str]], deadline: float) -> Plan:
        '''
        Expands *into* one unit's orders at a time - keeps + expands the best beam_width number of partial plans at each step.
        Returns the best complete plan reached pre-deadline.
        '''
        beam_width = self._feature('beam_width', self.beam_width)
        beam: list[tuple[float, Plan]] = [(0.0, {})] # (score, Plan)

        for location in self._beam_location_sequence(moveable_locations):
            if time.perf_counter() > deadline:
                break

            candidate_orders = self._candidate_orders(location, legal_orders)
            if not candidate_orders:
                continue

            expanded_plans: list[tuple[float, Plan]] = [] # All combinations of 1 beam partial plan + 1 new order from this location
            for _, plan in beam:
                for order in candidate_orders:
                    if self._conflicts_with_plan(plan, order):
                        continue
                    extended_plan = {**plan, location: order}
                    expanded_plans.append((self._score_plan(extended_plan), extended_plan))

            if expanded_plans: # Incase every candidate order conflicts with every beam plan, beam isn't reassigned
                expanded_plans.sort(key=lambda scored_plan: scored_plan[0], reverse=True)
                beam = expanded_plans[:beam_width]

        return max(beam, key=lambda scored_plan: scored_plan[0])[1]


    def _beam_location_sequence(self, moveable_locations: list[str]) -> list[str]:
        '''
        Returns the moveable locations in the order the beam decides them: each unit followed by neighbours.
        This way, a support order is decided just after the unit it may support.
        Now, sorted so the same seed replays the same game instead of arbitrary popping.
        '''
        ordered = []
        remaining = sorted(moveable_locations)
        while remaining:
            anchor = remaining.pop(0)
            ordered.append(anchor)
            # Decide own units next door right after their anchor
            neighbours = [neighbour for neighbour in sorted(self.map_graph.get(anchor, ())) if neighbour in remaining]
            for neighbour in neighbours:
                remaining.remove(neighbour)
                ordered.append(neighbour)

        return ordered


    def _candidate_orders(self, location: str, legal_orders: dict[str, list[str]]) -> list[str]:
        '''
        Returns the orders worth considering for a single own unit: legal, plausibly useful, pruned down by _prune_candidate_orders.
        '''
        candidate_orders = []
        for order in legal_orders.get(location, []):
            parsed_order = parse_order(order)
            if parsed_order is None:
                continue
            # Supporting or convoying a unit that is not ours is legal (teaming) but generally not what we want
            if parsed_order.kind in (SUPPORT, CONVOY) and self.occupied_by.get(parsed_order.supported_location) != self.power_name:
                continue
            candidate_orders.append(order)
        candidate_orders.sort()

        return self._prune_candidate_orders(candidate_orders)


    def _conflicts_with_plan(self, plan: Plan, order: str) -> bool:
        '''
        Returns True if the order clashes with orders already in the plan:
            - two units sent to the same province (bounce), or
            - move into a province another own unit already occupies.
        '''
        parsed_order = parse_order(order)
        if parsed_order is None or parsed_order.kind != MOVE:
            return False

        # An order exists in the plan for another own unit to move there
        for existing_order in plan.values():
            parsed_existing_order = parse_order(existing_order)
            if parsed_existing_order is not None and parsed_existing_order.kind == MOVE and parsed_existing_order.destination == parsed_order.destination:
                return True

        # An own unit is already there and no order exists in the plan for it to leave
        if self.occupied_by.get(parsed_order.destination) == self.power_name:
            occupant_order = plan.get(parsed_order.destination, '')
            parsed_occupant_order = parse_order(occupant_order)
            if parsed_occupant_order is None or parsed_occupant_order.kind != MOVE:
                return True

        return False


    def _prune_candidate_orders(self, candidate_orders: list[str]) -> list[str]:
        '''
        Returns the candidate orders cut down to candidate_size, keeping the highest priority by _prune_priority.
        The sort is stable, so equal priorities keep the (sorted) order they came in - deterministic.
        '''
        if len(candidate_orders) <= self.candidate_size:
            return candidate_orders

        prioritised_orders = sorted(candidate_orders, key=self._prune_priority, reverse=True)
        return prioritised_orders[:self.candidate_size]


    def _prune_priority(self, order: str) -> float:
        '''
        Returns a cheap priority for keeping an order as a candidate, by what the order does (see prune_priorities).
        '''
        parsed_order = parse_order(order)
        if parsed_order is None:
            return 0.0

        if parsed_order.kind == MOVE:
            is_centre = parsed_order.destination in self.all_centres
            occupant = self.occupied_by.get(parsed_order.destination)

            if is_centre and occupant is None:
                return self.prune_priorities['empty_centre']
            if is_centre and occupant != self.power_name:
                return self.prune_priorities['enemy_unit_centre']
            if occupant is None:
                return self.prune_priorities['empty_province']
            if occupant == self.power_name:
                return self.prune_priorities['own_unit_province']
            return self.prune_priorities['enemy_unit_province']

        if parsed_order.kind == SUPPORT:
            return self.prune_priorities['support']
        if parsed_order.kind == HOLD:
            return self.prune_priorities['hold']
        return 0.0



##### Opponent Model #####

    def _observe_opponents(self, all_power_orders: dict[str, list[str]]):
        '''
        Counts for each opponent unit, the chance to do each thing and whether it actually did:
            - defence: the unit could move into a province another one of its own units stands on, and supports its hold,
            - attack: the unit could move into our ground (an own centre or a province our unit is on), and moves into it.
        Could move => could move along that unit type's own routes. A unit with no order just holds.
        Iterates this power's units rather than its orders, since the orders may not cover every unit.
        Such is the need for province_of_unit and type_of_unit rather than keying parsed_order.
        '''
        own_locations = set(self.game.get_centers(self.power_name)) | {province_of_unit(unit) for unit in self.game.get_units(self.power_name)}

        for power, counts in self.opponent_counts.items():
            parsed_plan = {} # (ordered) province : its ParsedOrder - this power's orders... keyed like a Plan
            for order in all_power_orders.get(power, []):
                parsed_order = parse_order(order)
                if parsed_order is not None:
                    parsed_plan[parsed_order.location] = parsed_order

            units = {province_of_unit(unit): type_of_unit(unit) for unit in self.game.get_units(power)} # province : type of this power's unit there
            for location, unit_type in sorted(units.items()):
                parsed_order = parsed_plan.get(location)
                reachable = self._graph_of(unit_type).get(location, set()) # provinces this unit could move into (defend/attack)

                if reachable & set(units): # Could support one of its own units to hold
                    counts['defence_opportunities'] += 1
                    if parsed_order is not None and parsed_order.kind == SUPPORT and parsed_order.destination is None:
                        counts['defences'] += 1

                if reachable & own_locations: # Could attack our ground
                    counts['attack_opportunities'] += 1
                    if parsed_order is not None and parsed_order.kind == MOVE and parsed_order.destination in own_locations:
                        counts['attacks'] += 1


    def _rate(self, num_done: int, num_opportunities: int, prior: float) -> float:
        '''
        Returns num_done / num_opportunities, pulled towards the prior while there are few opportunities observed.
        With the opponent model off, returns the prior.
        '''
        if not self._feature('opponent_model', True):
            return prior
        return (num_done + prior * self.num_prior_observations) / (num_opportunities + self.num_prior_observations)


    def _defence_rate(self, power: str) -> float:
        '''
        Returns how often this power's units support a hold when they could (one of its own units is within reach).
        '''
        counts = self.opponent_counts[power]
        return self._rate(counts['defences'], counts['defence_opportunities'], self.rate_priors['defence'])


    def _attack_rate(self, power: str) -> float:
        '''
        Returns how often this power's units move into our ground when they could (one of our units or centres is within reach).
        '''
        counts = self.opponent_counts[power]
        return self._rate(counts['attacks'], counts['attack_opportunities'], self.rate_priors['attack'])



##### Scoring Heuristic #####

    def _enemy_defence_strength(self, location: str) -> float:
        '''
        Returns the estimated defensive strength the occupant of a province could muster in holding:
            - 0 if empty or ours.
            - 1 for the unit itself + that power's defence rate per unit of the same power that could move in (support the hold).
        '''
        occupant = self.occupied_by.get(location)

        if occupant is None or occupant == self.power_name:
            return 0.0

        if not self._feature('enemy_defence_supports', True):
            return 1.0

        defence_strength = 1.0
        defence_rate = self._defence_rate(occupant)

        for neighbour in self.map_graph.get(location, ()):
            occupant_neighbour = self.occupied_by.get(neighbour)
            if occupant_neighbour == occupant and self._can_reach(neighbour, location):
                defence_strength += defence_rate

        return defence_strength


    def _enemy_attack_chance(self, location: str) -> float:
        '''
        Returns the chance an own province is attacked: the highest attack rate among enemy units that could move in, 0 if none.
        TODO: alter this to mirror _enemy_defence_strength and change comparison calculations in scoring.
        '''
        attack_chance = 0.0
        for neighbour in self.map_graph.get(location, ()):
            occupant_neighbour = self.occupied_by.get(neighbour)
            if occupant_neighbour is not None and occupant_neighbour != self.power_name and self._can_reach(neighbour, location):
                attack_chance = max(attack_chance, self._attack_rate(occupant_neighbour))

        return attack_chance


    def _score_plan(self, plan: Plan) -> float:
        '''
        Returns the score of the plan - two steps:
            - predict what the plan achieves,
            - judge the outcome of the plan
        Then the beam search evaluates/compares the returned scores.
        The two steps are split apart so the weights describe what we want rather than what the rules do.
        '''
        return self._evaluate_outcome(self._predict_outcome(plan))


    def _predict_outcome(self, plan: Plan) -> PredictedOutcome:
        '''
        Returns a rough adjudication of a plan, ignoring what every enemy power does this turn.
        Only separates a plan that takes ground/makes progress from one that walks into a wall.
        '''
        parsed_plan = {location: parse_order(order) for location, order in plan.items()} # (ordered) province : its ParsedOrder

        parsed_move_orders: list[ParsedOrder] = [] # Every plan order that is a move
        parsed_support_orders: list[ParsedOrder] = [] # Every plan order that is a support
        for parsed_order in parsed_plan.values():
            if parsed_order is None:
                continue
            if parsed_order.kind == MOVE:
                parsed_move_orders.append(parsed_order)
            elif parsed_order.kind == SUPPORT:
                parsed_support_orders.append(parsed_order)

        move_support_at: dict[str, int] = defaultdict(int) # (destination) province : count of our supports for a move into it
        hold_support_at: dict[str, int] = defaultdict(int) # (own unit) province : count of our supports for the unit holding there
        num_wasted_supports = 0

        # A support only counts if the unit it names is really doing what the support says
        # A unit with no order in the plan yet may still be ordered to move, so uncounted (continue)
        for parsed_support_order in parsed_support_orders:
            parsed_supported_unit_order = parsed_plan.get(parsed_support_order.supported_location)
            if parsed_supported_unit_order is None:
                continue

            if parsed_support_order.destination is not None: # Move support: check that the supported unit is ordered into the same province
                if (parsed_supported_unit_order.kind == MOVE) and (parsed_supported_unit_order.destination == parsed_support_order.destination):
                    move_support_at[parsed_support_order.destination] += 1
                else:
                    num_wasted_supports += 1
            elif parsed_supported_unit_order.kind != MOVE: # Hold support confirmed: supporter is supporting a hold and supported is not moving
                hold_support_at[parsed_support_order.supported_location] += 1
            else:
                num_wasted_supports += 1

        # Ablation flag for the agent to see that its supports help
        use_support = self._feature('coordinated_support', True)

        # For final units: start with every ordered unit, take away moving units (where they end up is still in contest)
        final_units = {location: parsed_order.unit_type for location, parsed_order in parsed_plan.items()}
        for parsed_move_order in parsed_move_orders:
            del final_units[parsed_move_order.location]
        dislodged_locations: set[str] = set()
        num_failed_attacks = 0

        for parsed_move_order in parsed_move_orders:
            destination = parsed_move_order.destination # Re-used many times
            attack_strength = 1 + (int(use_support) * move_support_at.get(destination, 0))

            if attack_strength <= self.enemy_defence_strength_at.get(destination, 0.0): # Predicted to bounce: unit stays where it was
                num_failed_attacks += 1
                final_units[parsed_move_order.location] = parsed_move_order.unit_type
                continue

            final_units[destination] = parsed_move_order.unit_type # Successful move
            occupant = self.occupied_by.get(destination)
            if occupant is not None and occupant != self.power_name: # Successfully dislodged an enemy unit
                dislodged_locations.add(destination)

        return PredictedOutcome(final_units, dislodged_locations, num_failed_attacks, num_wasted_supports, hold_support_at)


    def _evaluate_outcome(self, outcome: PredictedOutcome) -> float:
        '''
        Returns the score of a predicted outcome (of a plan), weighted by:
            - Centres taken and kept,
            - Ground gained towards the next centre,
            - Units lined up next to enemy-occupied target centres (pressure),
            - Enemies pushed (dislodged) off it,
            - Wasted plan unit turns,
            - Own centres left empty or supported, times the chance each is attacked (to be changed, see TODO in _enemy_attack_chance)
        Sorted sets before walking through for determinism (due to float errors)
        '''
        use_progress = self._feature('distance_progress', True)
        use_defence = self._feature('defence', True)
        use_season_capture = self._feature('season_capture', True)
        use_pressure = self._feature('pressure', True)

        # A centre changes ownership only at the end of a Fall turn, so standing on one through Spring is occupy not capture
        target_centre_weight = self.score_weights['capture_centre']
        if not self.is_fall and use_season_capture:
            target_centre_weight = self.score_weights['occupy_centre']

        score = 0.0
        num_attackers_at: dict[str, int] = defaultdict(int) # (enemy-occupied target) centre : own units next to it

        for location, unit_type in sorted(outcome.final_units.items()):
            if location in self.own_centres:
                score += self.score_weights['keep_centre']
            elif location in self.all_centres:
                score += target_centre_weight

            if use_progress:
                # Leaving one of our own centres towards the next centre we want is a gain
                # Use one of the bfs distance maps to measure so
                distance = min(self._distance_to_target(location, unit_type), MAX_DISTANCE)
                score += self.score_weights['progress'] * (MAX_DISTANCE - distance)

            if use_pressure:
                # Next to an enemy-occupied target centre: counts only up to the number of attackers needed to dislodge it
                for centre in self._pressure_centres(location, unit_type):
                    num_attackers_at[centre] += 1
                    if num_attackers_at[centre] <= self.attackers_needed_at[centre]:
                        score += self.score_weights['pressure']

        score += self.score_weights['dislodge_unit'] * len(outcome.dislodged_locations)
        score += self.score_weights['wasted_support'] * outcome.num_wasted_supports
        score += self.score_weights['failed_attack'] * outcome.num_failed_attacks

        if use_defence:
            # Attack chance is learned per enemy power (was yes/no), so a quiet neighbour is not guarded against.
            # TODO: next step is to count, per centre, how many units the strongest individual enemy power can attack it with
            for centre in sorted(self.own_centres):
                if centre not in outcome.final_units: # Walked away from the centre
                    score += self.score_weights['expose_centre'] * self.enemy_attack_chance_at[centre]
                elif outcome.hold_support_at.get(centre, 0):
                    score += self.score_weights['hold_support'] * self.enemy_attack_chance_at[centre]

        return score