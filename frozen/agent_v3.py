from collections import defaultdict, deque
from typing import NamedTuple
import time
import timeout_decorator
from agent_baselines import Agent

# Order 'kinds', as appearing in the order string (movement has kind '-')
MOVE, RETREAT, SUPPORT, CONVOY, HOLD, BUILD, DISBAND = '-', 'R', 'S', 'C', 'H', 'B', 'D'

# Type alias for plans... (own unit) province : the order string which that unit will be given
Plan = dict[str, str]

# Assigned distance for provinces with no route to target centre, + distance cap on otherwise dominating stranded units
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
    def __init__(self, agent_name: str='TestBeamAgent'):
        super().__init__(agent_name)

        self.debug = False

        self.time_budget = 0.6 # Set well under one-second for safety

        # Agent parameters
        self.beam_width = 16 # = 1 makes this greedy search (possible 'improvement' greedy -> beam)
        self.candidate_size = 16 # legal orders considered per unit

        # Heuristic weights, term name : multiplier. Tuned by experiment, see results/.
        self.weights: dict[str, float] = {
            'capture_centre': 30.0,   # end a Fall turn on a centre we do not own - ownership only changes then
            'occupy_centre' :  8.0,   # end a Spring turn on one - no capture yet, but in place for the Fall
            'hold_centre'   :  2.0,   # end the turn on an own centre
            'progress'      :  3.0,   # per hop closer to the nearest target centre, along that unit type's routes
            'dislodge_unit' :  4.0,   # predicted to push an enemy unit off the province it stands on
            'failed_attack' : -2.0,   # predicted bounce: a wasted unit-turn
            'wasted_support': -2.0,   # support for a move or hold the plan does not actually make
            'expose_centre' : -6.0,   # an own centre left empty with an enemy next door
            'support_hold'  :  2.0,   # supporting an own unit sitting on a threatened own centre
            'pressure'      :  6.0    # per own unit next to a target centre an enemy unit stands on, up to the number needed to dislodge
        }

        # Retreat destination priorities, feature of the destination : priority. Highest is taken.
        self.retreat_priors: dict[str, float] = {
            'home_centre'        : 10.0,   # retreat onto one of our home centres
            'other_centre'       :  8.0,   # retreat onto any other centre
            'other_province'     :  2.0,   # retreat onto a non-centre province
            'enemy_neighbour'    :  -1.5,  # per enemy unit next to the destination
            'next_to_home_centre':  1.0    # destination (not home centre) borders one of our home centres
        }

        # Candidate pruning priorities, what the order does : priority. Higher is kept first.
        self.prune_priors: dict[str, float] = {
            'empty_centre'       : 10.0,   # move into an empty centre, whoever owns it
            'enemy_unit_centre'  :  8.0,   # move into a centre an enemy unit stands on
            'empty_province'     :  3.0,   # move into an empty non-centre province
            'own_unit_province'  : -5.0,   # move into a province one of our own units stands on
            'enemy_unit_province':  4.0,   # move into a non-centre province an enemy unit stands on
            'support'            :  5.0,   # any support, for a move or a hold
            'hold'               :  1.0    # hold
        }

        self.is_fall = False # True in a Fall phase (movement or retreat) - the only time when centre ownership changes
        self.occupied_by: dict[str, str] = {} # (occupied) province : power whose unit stands there
        self.centre_owned_by: dict[str, str] = {} # centre : power owning it, absent if unowned
        self.army_distance_to_target: dict[str, int] = {} # (army-reachable) province : hops to nearest target centre along army routes only
        self.fleet_distance_to_target: dict[str, int] = {} # (fleet-reachable) province : hops to nearest target centre along fleet routes only
        self.units_needed_at: dict[str, int] = {} # (target centre with enemy unit on it) : own units needed next to it to dislodge the enemy unit
        self.army_pressure_centres: dict[str, list[str]] = {} # (army) province : [target centres an enemy unit stands on that an army there could move into]
        self.fleet_pressure_centres: dict[str, list[str]] = {} # (fleet) province : [target centres an enemy unit stands on that a fleet there could move into]
        self.defence_at: dict[str, float] = {} # (all) province : defence strength of the enemy unit there, 0 if empty or ours


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


    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        '''
        End of a phase: replays every power's orders onto our own board copy,
        to keep in step with the real game.
        '''
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

        # Following is built once per turn here rather than hundreds of time per turn in _score_plan

        self.defence_at = {location: self._defence_at(location) for location in self.map_graph}
        # Smallest attack strength that beats the defence (a tie bounces): one attacker + supports
        self.units_needed_at = {centre: int(self.defence_at[centre]) + 1 for centre in self.target_centres if self.defence_at.get(centre, 0.0) > 0}
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

        if not adjustable_locations:
            return []

        self._read_board()

        # Score every legal build
        scored_builds: list[tuple[int, str, str]] = [] # (hops to nearest target centre, build order, home centre)
        scored_disbands = [] # see TODO
        for location in sorted(adjustable_locations):
            for order in legal_orders.get(location, []):
                parsed_order = parse_order(order)
                if parsed_order and parsed_order.kind == BUILD:
                    distance = self._distance_to_target(parsed_order.location, parsed_order.unit_type)
                    scored_builds.append((distance, order, parsed_order.location))
                elif parsed_order and parsed_order.kind == DISBAND:
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
        for location in retreatable_locations:
            retreat_options = []
            for order in legal_orders.get(location, []):
                parsed_order = parse_order(order)
                if parsed_order and parsed_order.kind == RETREAT:
                    retreat_options.append(order)

            if not retreat_options:
                continue

            best_order, best_score = None, -float('inf')
            for order in sorted(retreat_options): # Sorted so ties break the same way every run
                destination = parse_order(order).destination
                score = 0.0

                if destination in self.home_centres:
                    score += self.retreat_priors['home_centre']
                elif destination in self.all_centres:
                    score += self.retreat_priors['other_centre']
                else:
                    score += self.retreat_priors['other_province']

                num_enemy_neighbours = 0
                for neighbour in self.map_graph.get(destination, ()):
                    occupant = self.occupied_by.get(neighbour)
                    if occupant is not None and occupant != self.power_name:
                        num_enemy_neighbours += 1
                score += num_enemy_neighbours * self.retreat_priors['enemy_neighbour']

                if destination not in self.home_centres and self.map_graph.get(destination, set()) & self.home_centres:
                    score += self.retreat_priors['next_to_home_centre']

                if score > best_score:
                    best_order, best_score = order, score

            if best_order:
                orders.append(best_order)

        return orders



##### Board and Map #####

    def _read_board(self):
        '''
        Initialises relevant data structures, reads the board at the start of every phase (movement, retreat, adjustment) - everything shared between phases:
            - who stands where (occupied_by), 
            - who owns which centre (own_centres, centres_owned_by),
            - the season (is_fall),
            - which centres we are after (target_centres) and how far each unit type is from one.
        '''
        self.occupied_by = self._build_occupied_by()
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


    def _pressure_centres(self, location: str, unit_type: str) -> list[str]:
        '''
        Returns the target centres with an enemy unit on them that a unit of this type ('A' or 'F') on this province could move into,
        or [] if none.
        '''
        if unit_type == 'A':
            return self.army_pressure_centres.get(location, [])
        return self.fleet_pressure_centres.get(location, [])


    def _build_centre_owned_by(self) -> dict[str, str]:
        '''
        Returns centre: power controlling it.
        '''
        return {centre: power for power in self.game.powers for centre in self.game.get_centers(power)}


    def _build_occupied_by(self) -> dict[str, str]:
        '''
        Returns province : power holding unit there.
        '''
        occupied_by = {}
        for power in self.game.powers:
            for unit in self.game.get_units(power):
                location = province_of(unit.split()[1].lstrip('*'))
                occupied_by[location] = power

        return occupied_by


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
                if (self.game.map.abuts(unit_type, source, '-', target)):
                    graph[province_of(source)].add(province_of(target))

        return graph


    def _build_pressure_centres(self, graph: dict[str, set[str]]) -> dict[str, list[str]]:
        '''
        Returns province (next to target centre) : [target centres an enemy unit stands on reachable in one move along graph's edges].
        This is what attracts units to support attacks on enemy centres where needed.
        '''
        pressure_centres = {}
        for location, neighbours in sorted(graph.items()):
            centres = [neighbour for neighbour in sorted(neighbours) if neighbour in self.units_needed_at]
            if centres:
                pressure_centres[location] = centres

        return pressure_centres


    def _build_distance_to(self, sources: list[str], graph: dict[str, set[str]]) -> dict[str, int]:
        '''
        Returns (reachable) provinces : hops to nearest source of the sources, moving only along the graph's edges.
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
            if parsed_existing_order and parsed_existing_order.kind == MOVE and parsed_existing_order.destination == parsed_order.destination:
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
        Returns a cheap priority for keeping an order as a candidate, by what the order does (see prune_priors).
        '''
        parsed_order = parse_order(order)
        if parsed_order is None:
            return 0.0

        if parsed_order.kind == MOVE:
            is_centre = parsed_order.destination in self.all_centres
            occupant = self.occupied_by.get(parsed_order.destination)

            if is_centre and occupant is None:
                return self.prune_priors['empty_centre']
            if is_centre and occupant != self.power_name:
                return self.prune_priors['enemy_unit_centre']
            if occupant is None:
                return self.prune_priors['empty_province']
            if occupant == self.power_name:
                return self.prune_priors['own_unit_province']
            return self.prune_priors['enemy_unit_province']

        if parsed_order.kind == SUPPORT:
            return self.prune_priors['support']
        if parsed_order.kind == HOLD:
            return self.prune_priors['hold']
        return 0.0



##### Scoring Heuristic ######

    def _defence_at(self, location: str) -> float:
        '''
        Returns the strength the occupant of a province could muster in holding:
            - 0 if empty or ours.
            - 1 for the unit + a fraction per neighbouring unit of the same power (could support hold).
        '''
        occupant = self.occupied_by.get(location)
        if occupant is None or occupant == self.power_name:
            return 0.0

        if not self._feature('opponent_defence_estimate', True):
            return 1.0

        defence_strength = 1.0
        for neighbour in self.map_graph.get(location, ()):
            if self.occupied_by.get(neighbour) == occupant:
                defence_strength += 0.5

        return defence_strength


    def _is_threatened(self, location: str) -> bool:
        '''
        Returns True if this province is bordered by any enemy unit (could be attacked).
        '''
        for neighbour in self.map_graph.get(location, ()):
            occupant = self.occupied_by.get(neighbour)
            if occupant is not None and occupant != self.power_name:
                return True

        return False


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
        parsed_plan = {location: parse_order(order) for (location, order) in plan.items()} # (ordered) province : its ParsedOrder

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

            if attack_strength <= self.defence_at.get(destination, 0.0): # Predicted to bounce: unit stays where it was
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
            - Wasted plan unit turns.
        Sorted sets before walking through for determinism (due to float errors)
        '''
        use_progress = self._feature('distance_progress', True)
        use_defence = self._feature('defence', True)
        use_season_capture = self._feature('season_capture', True)
        use_pressure = self._feature('pressure', True)

        # A centre changes ownership only at the end of a Fall turn, so standing on one through Spring is occupy not capture
        target_centre_weight = self.weights['capture_centre']
        if not self.is_fall and use_season_capture:
            target_centre_weight = self.weights['occupy_centre']
        
        score = 0.0
        num_units_next_to: dict[str, int] = defaultdict(int) # target centre (with an enemy unit on it) : own units next to it

        for location, unit_type in sorted(outcome.final_units.items()):
            if location in self.own_centres:
                score += self.weights['hold_centre']
            elif location in self.all_centres:
                score += target_centre_weight

            if use_progress:
                # Leaving one of our own centres towards the next centre we want is a gain
                # Use one of the bfs distance maps to measure so
                distance = min(self._distance_to_target(location, unit_type), MAX_DISTANCE)
                score += self.weights['progress'] * (MAX_DISTANCE - distance)

            if use_pressure:
                # Next to a target centre an enemy unit stands on: counts only up to however many units needed to dislodge it
                for centre in self._pressure_centres(location, unit_type):
                    num_units_next_to[centre] += 1
                    if num_units_next_to[centre] <= self.units_needed_at[centre]:
                        score += self.weights['pressure']
        
        score += self.weights['dislodge_unit'] * len(outcome.dislodged_locations)
        score += self.weights['wasted_support'] * outcome.num_wasted_supports
        score += self.weights['failed_attack'] * outcome.num_failed_attacks

        if use_defence:
            # TODO: currently a yes/no threat test. 
            # Next step is to count, per centre, how many units the strongest individual enemy power can attack it with
            # VERSUS how many of ours could support it or bounce them out... same comparison from attacking side, support only stands if no cuts
            for centre in sorted(self.own_centres):
                if centre not in outcome.final_units: # Walked away from the centre
                    if self._is_threatened(centre):
                        score += self.weights['expose_centre']
                elif self._is_threatened(centre) and outcome.hold_support_at.get(centre, 0):
                    score += self.weights['support_hold']

        return score