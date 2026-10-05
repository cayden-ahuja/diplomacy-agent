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
        is_convoyed: True for an army's move by convoy ('...VIA'), which only gets there if a fleet convoys it.
    '''
    unit_type         : str
    location          : str
    kind              : str
    destination       : str | None
    supported_location: str | None
    is_convoyed       : bool = False


def parse_order(order: str) -> ParsedOrder | None:
    '''
    'F NTH S A EDI - YOR' -> ParsedOrder('F', 'NTH', 'S', 'YOR', 'EDI') | None for 'WAIVE'.
    '''
    parts = order.split()
    if len(parts) < 3: # Waive
        return None

    unit_type, location, kind = parts[0], province_of(parts[1]), parts[2]

    # Move or Retreat:
    #   'F IRI - MAO' (move),
    #   'A WAL R LON' (retreat)
    #   'A EDI - NWY VIA' (move by convoy)
    if kind in (MOVE, RETREAT):
        return ParsedOrder(unit_type, location, kind, province_of(parts[3]), None, parts[-1] == 'VIA')

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
        num_wasted_supports: our supports (and convoys) for a move or hold the plan does not actually make.
    '''
    final_units        : dict[str, str]
    dislodged_locations: set[str]
    num_failed_attacks : int
    num_wasted_supports: int



class StudentAgent(Agent):
    '''
    Beam search agent: builds a plan (keyed set of orders) one unit at a time,
    keeping the best beam_width partial plans at each step via the heuristic in _score_plan.
    '''
    @timeout_decorator.timeout(1)
    def __init__(self, agent_name: str='TestBeamAgent'):
        super().__init__(agent_name)

        self.debug = False

        self.time_budget = 0.6 # Set well under one-second for safety

        # Agent parameters
        self.beam_width = 16 # = 1 makes this greedy search (possible 'improvement' greedy -> beam)

        # Score weights, term name : multiplier in _evaluate_outcome. Tuned by experiment, see results/.
        self.score_weights: dict[str, float] = {
            'capture_centre': 30.0,   # end a Fall turn on a centre we do not own - ownership only changes then
            'occupy_centre' :  8.0,   # end a Spring turn on one - no capture yet, but in place for the Fall
            'keep_centre'   :  2.0,   # end the turn on an own centre
            'progress'      :  3.0,   # per hop closer to the nearest target centre, along that unit type's routes
            'dislodge_unit' :  4.0,   # predicted to push an enemy unit off the province it stands on
            'failed_attack' : -2.0,   # predicted bounce: a wasted unit-turn
            'wasted_support': -2.0,   # support (or convoy) for a move or hold the plan does not actually make
            'pressure'      :  6.0,   # per attacker next to a defended target centre, up to attackers_needed_at
            'garrison'      :  4.0    # per defender on or next to an attackable own centre, up to defenders_needed_at
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
            'defence': 0.5,    # a unit that could defend (support a hold or move back into an empty centre) does - even guess until shown otherwise
            'attack' : 1.0     # a unit that could move into our ground does - assumed until it shows otherwise
        }

        self.num_prior_observations: int = 3   # pretend observations behind each rate prior, so one early turn does not swing a rate
        self.opponent_counts: dict[str, dict[str, int]] = {}  # power : {count name : count}, over every movement phase so far

        self.is_fall: bool = False  # True in a Fall phase (movement or retreat) - the only time when centre ownership changes

        self.occupied_by              : dict[str, str] = {}     # (occupied) province : power whose unit stands there
        self.unit_type_at             : dict[str, str] = {}     # (occupied) province : type of unit there, 'A' or 'F'
        self.centre_owned_by          : dict[str, str] = {}     # (owned) centre : power owning it, absent if unowned
        self.army_distance_to_target  : dict[str, int] = {}     # (army-reachable) province : hops to nearest target centre along army routes only
        self.fleet_distance_to_target : dict[str, int] = {}     # (fleet-reachable) province : hops to nearest target centre along fleet routes only

        # Attacking their centres
        self.enemy_defence_strength_at: dict[str, float]     = {}  # (all) province : defence strength an enemy power could keep it with, 0 if ours, or empty and not an enemy centre
        self.attackers_needed_at      : dict[str, float]     = {}  # (defended target) centre : own units needed next to it to take it (a share of one while its defence is uncertain)
        self.army_pressure_centres    : dict[str, list[str]] = {}  # (army) province : [defended target centres that an army there could move into]
        self.fleet_pressure_centres   : dict[str, list[str]] = {}  # (fleet) province : [defended target centres that a fleet there could move into]

        # Defending our centres
        self.enemy_attack_strength_at: dict[str, float]     = {}  # (own) centre : attack strength the strongest enemy power could bring against it, 0 if none
        self.defenders_needed_at     : dict[str, float]     = {}  # (attackable own) centre : own units needed on or next to it to hold it (a share of one while the attack is uncertain)
        self.army_garrison_centres   : dict[str, list[str]] = {}  # (army) province : [attackable own centres that an army there could hold or move into]
        self.fleet_garrison_centres  : dict[str, list[str]] = {}  # (fleet) province : [attackable own centres that a fleet there could hold or move into]


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
        Spring/Fall movement phase: calling beam search builds a plan, then coordinate ascent improves it until time runs out.
        '''
        deadline = time.perf_counter() + self.time_budget

        legal_orders = self.game.get_all_possible_orders() # province : [legal order strings]
        moveable_locations = self.game.get_orderable_locations(self.power_name)

        if not moveable_locations: # Early exit
            return []

        self._read_board()

        # The following is built once per turn here rather than hundreds of times per turn in _score_plan

        # Attacking their centres: strength each enemy power could defend each with, from learned defence rates
        self.enemy_defence_strength_at = {location: self._enemy_defence_strength(location) for location in self.map_graph}
        # Attackers that beat it (a tie bounces the attacker, so one more than its defence strength): one moving unit + supporters
        self.attackers_needed_at = {centre: self.enemy_defence_strength_at[centre] + 1 for centre in self.target_centres if self.enemy_defence_strength_at.get(centre, 0.0) > 0}
        # Pressure centres adjacent per location
        self.army_pressure_centres = self._build_pressure_centres(self.army_graph)
        self.fleet_pressure_centres = self._build_pressure_centres(self.fleet_graph)


        # Defending our centres: strength the strongest enemy power could attack each with, from learned attack rates
        self.enemy_attack_strength_at = {centre: self._enemy_attack_strength(centre) for centre in self.own_centres}
        # Defenders that hold against it (a tie holds for the defender): the unit on the centre + hold supporters, as many as the attack strength
        self.defenders_needed_at = {centre: self.enemy_attack_strength_at[centre] for centre in self.own_centres if self.enemy_attack_strength_at.get(centre, 0.0) > 0}
        # Garrison centres adjacent per location
        self.army_garrison_centres = self._build_garrison_centres(self.army_graph)
        self.fleet_garrison_centres = self._build_garrison_centres(self.fleet_graph)

        # Every unit's candidate orders, worked out once for both search steps: beam + coordinate ascent
        candidate_orders_at = {location: self._candidate_orders(location, legal_orders) for location in sorted(moveable_locations)}

        plan = self._beam_search(moveable_locations, candidate_orders_at, deadline)
        if self._feature('coordinate_ascent', True):
            plan = self._coordinate_ascent(plan, candidate_orders_at, deadline)

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
        Returns the defended target centres that a unit of this type ('A' or 'F') on this province could move into,
        or [] if none.
        '''
        if unit_type == 'A':
            return self.army_pressure_centres.get(location, [])
        return self.fleet_pressure_centres.get(location, [])


    def _garrison_centres(self, location: str, unit_type: str) -> list[str]:
        '''
        Returns the attackable own centres that a unit of this type ('A' or 'F') on this province could hold or move into,
        or [] if none.
        '''
        if unit_type == 'A':
            return self.army_garrison_centres.get(location, [])
        return self.fleet_garrison_centres.get(location, [])


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
        Returns (province next to defended target centre) : [defended target centres reachable in one move along graph's edges].
        This is what attracts units to line up attacks on defended target centres where needed.
        '''
        pressure_centres = {}
        for location, neighbours in sorted(graph.items()):
            centres = [neighbour for neighbour in sorted(neighbours) if neighbour in self.attackers_needed_at]
            if centres:
                pressure_centres[location] = centres

        return pressure_centres


    def _build_garrison_centres(self, graph: dict[str, set[str]]) -> dict[str, list[str]]:
        '''
        Returns (province on or next to attackable own centre) : [attackable own centres it stands on or reaches in one move along graph's edges].
        This is what attracts units to line up defences on attackable own centres where needed.
        '''
        garrison_centres = {}
        for location, neighbours in sorted(graph.items()):
            # Standing on the centre holds it, standing next to it can support the hold or bounce the attacker off
            centres = [province for province in sorted(neighbours | {location}) if province in self.defenders_needed_at]
            if centres:
                garrison_centres[location] = centres

        return garrison_centres


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

    def _beam_search(self, moveable_locations: list[str], candidate_orders_at: dict[str, list[str]], deadline: float) -> Plan:
        '''
        Expands *into* one unit's orders at a time - keeps + expands the best beam_width number of partial plans at each step.
        Returns the best complete plan reached pre-deadline.
        '''
        beam_width = self._feature('beam_width', self.beam_width)
        beam: list[tuple[float, Plan]] = [(0.0, {})] # (score, Plan)

        for location in self._beam_location_sequence(moveable_locations):
            if time.perf_counter() > deadline:
                break

            candidate_orders = candidate_orders_at.get(location, [])
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


    def _coordinate_ascent(self, plan: Plan, candidate_orders_at: dict[str, list[str]], deadline: float) -> Plan:
        '''
        Improves the beam's best plan by (block) coordinate ascent - each unit's order is one coordinate of the plan.
        Put another way: the plan's score is a function of every unit's order, maximised one variable (or one block of variables) at a time.
        Repeats until no change helps or time runs out:
            - change one unit's order at a time, keeping any change that scores higher,
            - change a move and support for it together (two coordinates at once), which single order changes cannot find
              (the move alone may score lower until the support is in as well, and the support is wasted until the move is in).
        The beam decides each unit once, in a fixed sequence: coordinate ascent can revisit any unit once the whole plan is known and passed in.
        '''
        best_score = self._score_plan(plan)
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False

            # One unit at a time
            for location, candidate_orders in candidate_orders_at.items():
                for order in candidate_orders:
                    if order == plan.get(location):
                        continue
                    plan_remainder = {other: other_order for other, other_order in plan.items() if other != location}
                    if self._conflicts_with_plan(plan_remainder, order):
                        continue
                    trial_plan = {**plan_remainder, location: order}
                    trial_score = self._score_plan(trial_plan)
                    if trial_score > best_score:
                        best_score, plan = trial_score, trial_plan
                        improved = True

            # A move and a support for it together
            for location, candidate_orders in candidate_orders_at.items():
                for order in candidate_orders:
                    parsed_order = parse_order(order)
                    if parsed_order is None or parsed_order.kind != MOVE or time.perf_counter() > deadline:
                        continue
                    for supporter_location, supporter_orders in candidate_orders_at.items():
                        if supporter_location == location:
                            continue
                        support_order = self._support_order_for(supporter_orders, parsed_order)
                        if support_order is None:
                            continue
                        plan_remainder = {other: other_order for other, other_order in plan.items() if other not in (location, supporter_location)}
                        if self._conflicts_with_plan(plan_remainder, order) or self._conflicts_with_plan(plan_remainder, support_order):
                            continue
                        trial_plan = {**plan_remainder, location: order, supporter_location: support_order}
                        trial_score = self._score_plan(trial_plan)
                        if trial_score > best_score:
                            best_score, plan = trial_score, trial_plan
                            improved = True

        return plan


    def _support_order_for(self, orders: list[str], parsed_move_order: ParsedOrder) -> str | None:
        '''
        Returns the first of these orders that supports this move, or None if none do.
        '''
        for order in orders:
            parsed_order = parse_order(order)
            if parsed_order is not None and parsed_order.kind == SUPPORT:
                if parsed_order.supported_location == parsed_move_order.location and parsed_order.destination == parsed_move_order.destination:
                    return order

        return None


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
        Returns the orders worth considering for a single own unit: every legal order, except supports and convoys for units not ours.
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

        return candidate_orders


    def _conflicts_with_plan(self, plan: Plan, order: str) -> bool:
        '''
        Returns True if the order clashes with orders already in the plan:
            - two units sent to the same province (bounce),
            - move into a province another own unit already occupies, or
            - two own units swapping places (they bounce - only a convoy lets units swap)
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
            is_by_convoy = parsed_order.is_convoyed or parsed_occupant_order.is_convoyed
            if parsed_occupant_order.destination == parsed_order.location and not is_by_convoy: # Swapping places by land
                return True

        return False



##### Opponent Model #####

    def _observe_opponents(self, all_power_orders: dict[str, list[str]]):
        '''
        Counts for each opponent unit, the chance to do each thing and whether it actually did:
            - defence: the unit could move into a province another one of its own units stands on, and supports its hold,
                       or into one of its own empty centres, and moves into it,
            - attack: the unit could move into our ground (an own centre or a province our unit is on), and moves into it.
        Could move => could move along that unit type's own routes. A unit with no order just holds.
        Iterates this power's units rather than its orders, since the orders may not cover every unit.
        Such is the need for province_of_unit and type_of_unit rather than keying parsed_order.
        '''
        own_locations = set(self.game.get_centers(self.power_name)) | {province_of_unit(unit) for unit in self.game.get_units(self.power_name)}
        occupied_locations = set(self._build_occupied_by())

        for power, counts in self.opponent_counts.items():
            parsed_plan = {} # (ordered) province : its ParsedOrder - this power's orders... keyed like a Plan
            for order in all_power_orders.get(power, []):
                parsed_order = parse_order(order)
                if parsed_order is not None:
                    parsed_plan[parsed_order.location] = parsed_order

            units = {province_of_unit(unit): type_of_unit(unit) for unit in self.game.get_units(power)} # province : type of this power's unit there
            empty_centres = set(self.game.get_centers(power)) - occupied_locations # this power's centres with no unit on them
            for location, unit_type in sorted(units.items()):
                parsed_order = parsed_plan.get(location)
                reachable = self._graph_of(unit_type).get(location, set()) # provinces this unit could move into (defend/attack)

                if reachable & (set(units) | empty_centres): # Could support one of its own units to hold, or move into one of its own empty centres
                    counts['defence_opportunities'] += 1
                    if parsed_order is not None and parsed_order.kind == SUPPORT and parsed_order.destination is None:
                        counts['defences'] += 1
                    elif parsed_order is not None and parsed_order.kind == MOVE and parsed_order.destination in empty_centres:
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
        Returns how often this power's units defend when they could (support a hold, or move into one of its own empty centres).
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
        Returns the estimated defensive strength an enemy power could muster to keep a province:
            - 0 if ours, or empty and not an enemy centre.
            - 1 for its unit on it (0 for empty enemy centre) + that power's defence rate per unit of the same power that could move in
              (support hold, or move back into the empty centre and bounce off our attacker).
        '''
        occupant = self.occupied_by.get(location)
        defender = occupant if occupant is not None else self.centre_owned_by.get(location)

        if defender is None or defender == self.power_name:
            return 0.0

        defence_strength = 1.0 if occupant is not None else 0.0
        if not self._feature('enemy_defence_neighbours', True):
            return defence_strength

        defence_rate = self._defence_rate(defender)

        for neighbour in self.map_graph.get(location, ()):
            occupant_neighbour = self.occupied_by.get(neighbour)
            if occupant_neighbour == defender and self._can_reach(neighbour, location):
                defence_strength += defence_rate

        return defence_strength


    def _enemy_attack_strength(self, location: str) -> float:
        '''
        Returns the estimated attack strength the strongest enemy power could muster against an own province:
            - 0 if no enemy unit could move in.
            - that power's attack rate per unit of it that could move in (one moves, others support)
        '''
        attack_strength_of: dict[str, float] = defaultdict(float) # (enemy) power : its attack strength against this province
        for neighbour in self.map_graph.get(location, ()):
            occupant_neighbour = self.occupied_by.get(neighbour)
            if occupant_neighbour is not None and occupant_neighbour != self.power_name and self._can_reach(neighbour, location):
                attack_strength_of[occupant_neighbour] += self._attack_rate(occupant_neighbour)

        return max(attack_strength_of.values(), default=0.0)


    def _needed_share(self, num_units: int, num_needed: float) -> float:
        '''
        Returns how much of the num_units-th unit is still needed (defending and attacking): 1 while within the number needed, fractional remainder on the edge, 0 after.
        Both numbers needed are expected, from learned rates => last unit pays its share, scaling pressure and garrison.
        '''
        return min(1.0, max(0.0, num_needed - (num_units - 1)))


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

        parsed_move_orders   : list[ParsedOrder] = []  # Every plan order that is a move
        parsed_support_orders: list[ParsedOrder] = []  # Every plan order that is a support
        parsed_convoy_orders : list[ParsedOrder] = []  # Every plan order that is a convoy
        for parsed_order in parsed_plan.values():
            if parsed_order is None:
                continue
            if parsed_order.kind == MOVE:
                parsed_move_orders.append(parsed_order)
            elif parsed_order.kind == SUPPORT:
                parsed_support_orders.append(parsed_order)
            elif parsed_order.kind == CONVOY:
                parsed_convoy_orders.append(parsed_order)

        move_support_at: dict[str, int] = defaultdict(int) # (destination) province : count of our supports for a move into it
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
            elif parsed_supported_unit_order.kind == MOVE: # Hold support for a unit that is moving away
                num_wasted_supports += 1

        # A convoy only counts if the army it names is really moving where the convoy says
        convoyed_moves: set[tuple[str, str]] = set() # {(army province, destination)} one of our fleets convoys
        for parsed_convoy_order in parsed_convoy_orders:
            parsed_convoyed_unit_order = parsed_plan.get(parsed_convoy_order.supported_location)
            if parsed_convoyed_unit_order is None:
                continue

            if (parsed_convoyed_unit_order.kind == MOVE) and (parsed_convoyed_unit_order.destination == parsed_convoy_order.destination):
                convoyed_moves.add((parsed_convoy_order.supported_location, parsed_convoy_order.destination))
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

            is_unconvoyed = parsed_move_order.is_convoyed and (parsed_move_order.location, destination) not in convoyed_moves # A move by convoy with no fleet convoying goes nowhere
            if is_unconvoyed or attack_strength <= self.enemy_defence_strength_at.get(destination, 0.0): # Predicted to bounce: unit stays where it was
                num_failed_attacks += 1
                final_units[parsed_move_order.location] = parsed_move_order.unit_type
                continue

            final_units[destination] = parsed_move_order.unit_type # Successful move
            occupant = self.occupied_by.get(destination)
            if occupant is not None and occupant != self.power_name: # Successfully dislodged an enemy unit
                dislodged_locations.add(destination)

        return PredictedOutcome(final_units, dislodged_locations, num_failed_attacks, num_wasted_supports)


    def _evaluate_outcome(self, outcome: PredictedOutcome) -> float:
        '''
        Returns the score of a predicted outcome (of a plan), weighted by:
            - Centres taken and kept,
            - Ground gained towards the next centre,
            - Units lined up next to defended target centres (pressure),
            - Units lined up on or next to attackable own centres (garrison),
            - Enemies pushed (dislodged) off it,
            - Wasted plan unit turns.
        Sorted sets before walking through for determinism (due to float errors)
        '''
        use_progress = self._feature('distance_progress', True)
        use_season_capture = self._feature('season_capture', True)
        use_pressure = self._feature('pressure', True)
        use_garrison = self._feature('garrison', True)

        # A centre changes ownership only at the end of a Fall turn, so standing on one through Spring is occupy not capture
        target_centre_weight = self.score_weights['capture_centre']
        if not self.is_fall and use_season_capture:
            target_centre_weight = self.score_weights['occupy_centre']

        score = 0.0
        num_attackers_at: dict[str, int] = defaultdict(int) # (defended target) centre : own units next to it
        num_defenders_at: dict[str, int] = defaultdict(int) # (attackable own) centre : own units on or next to it

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
                # Next to a defended target centre: counts only up to the number of attackers needed to take it
                for centre in self._pressure_centres(location, unit_type):
                    num_attackers_at[centre] += 1
                    score += self.score_weights['pressure'] * self._needed_share(num_attackers_at[centre], self.attackers_needed_at[centre])

            if use_garrison:
                # On or next to an attackable own centre: counts only up to the number of defenders needed to hold it
                for centre in self._garrison_centres(location, unit_type):
                    num_defenders_at[centre] += 1
                    score += self.score_weights['garrison'] * self._needed_share(num_defenders_at[centre], self.defenders_needed_at[centre])

        score += self.score_weights['dislodge_unit'] * len(outcome.dislodged_locations)
        score += self.score_weights['wasted_support'] * outcome.num_wasted_supports
        score += self.score_weights['failed_attack'] * outcome.num_failed_attacks

        return score