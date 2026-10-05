import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
import traceback
import glob
import importlib
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from game import run_one_game
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent


# Module the 'default' variant loads (a frozen version is chosen per variant by '_agent' instead)
DEFAULT_AGENT = 'agent_30'

ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

RESULTS_DIR = 'results'
TIME_LIMIT = 1.0   # Seconds per agent call - going over disqualifies the agent
WIN_CENTRES = 18
END_YEAR = 1920
CENTRE_MILESTONES = (10, 14, 18)  # Record the first year we reach each of these


# Stand-in for the hidden agent: "a reasonably strong agent with a ~50% win rate in Scenario 2" (project info).
# Frozen agent_v4 (R3, opponent model) wins 51.8% of S2 games on average over seeds 3011/5011/7011/9011
# (60.0 / 55.7 / 41.4 / 50.0%); agent_v3b averages 45.0%, agent_v5 and later win well above 50%.
# Run from this folder (imports frozen/).
try:
    HIDDEN_STANDIN = importlib.import_module('frozen.agent_v4').StudentAgent
except ImportError:
    # frozen/ is not part of the submission, so fall back to the strongest provided baseline
    HIDDEN_STANDIN = GreedyAgent
    print('Note: frozen/agent_v4.py not found - Scenario 3 uses GreedyAgent as the hidden stand-in')


# Opponent pool for Scenarios 2 and 3 - random agent less likely to appear
S2_POOL = [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]


# Ablation variants, variant name : feature flags handed to the agent as self.features
# Grouped by the report section each one supports.
VARIANTS = {
    'default': {},

    # Basic technique: beam search (width 1 = greedy, default 16)
    'beam1' : {'beam_width': 1},
    'beam4' : {'beam_width': 4},
    'beam8' : {'beam_width': 8},
    'beam32': {'beam_width': 32},

    # Scoring heuristic terms
    'no_progress': {'distance_progress': False},
    'no_season'  : {'season_capture': False},
    'no_support' : {'coordinated_support': False},
    'no_pressure': {'pressure': False},
    'no_garrison': {'garrison': False},

    # New technique 1: learnt opponent model
    'no_opp_model'       : {'opponent_model': False},
    'no_enemy_neighbours': {'enemy_defence_neighbours': False},
    'no_opp_all'         : {'opponent_model': False, 'enemy_defence_neighbours': False},

    # New technique 2: coordinate ascent
    'no_coordinate_ascent': {'coordinate_ascent': False},

    # New technique 3: probabilistic outcome prediction
    'no_enemy_prediction': {'predict_repeat': False, 'predict_greedy': False},
    'no_support_cut'     : {'support_cut': False},
    'no_standoff'        : {'standoff': False},
    'no_dislodge_risk'   : {'dislodge_risk': False},
    'no_centre_loss'     : {'centre_loss': False},
    'no_block_build'     : {'block_build': False},
}


# Every frozen/agent_*.py becomes a variant automatically: agent_v3.py -> 'v3'.
# A variant can also combine a frozen agent with flags, e.g. {'_agent': 'frozen.agent_v3', 'beam_width': 1}
for path in sorted(glob.glob('frozen/agent_*.py')):
    module_name = os.path.basename(path)[:-3]                        # 'agent_v3'
    VARIANTS[module_name.replace('agent_', '')] = {'_agent': f'frozen.{module_name}'}


# Beam width sweep on the basic technique: frozen v1's scoring at other widths (v0 is width 1, v1 is width 16)
if os.path.exists('frozen/agent_v1.py'):
    for beam_width in (2, 4, 8, 32):
        VARIANTS[f'v1_beam{beam_width}'] = {'_agent': 'frozen.agent_v1', 'beam_width': beam_width}


# Marking rubric thresholds, scenario : [(win rate, mean centres, points), ...] - highest mark first
RUBRIC = {
    1: [(0.90, 16, 5), (0.20, 12, 3), (0.02, 7, 1)],
    2: [(0.50, 13, 5), (0.25, 10, 3), (0.02, 7, 1)],
    3: [(0.40, 12, 5), (0.20,  9, 3), (0.02, 7, 1)],
}



##### Phase Parsing #####

def parse_phase(phase):
    '''
    Returns the phase split into its parts: 'SPRING 1901 MOVEMENT' -> ('SPRING', 1901, 'MOVEMENT').
    'COMPLETED' -> ('', None, 'COMPLETED').
    '''
    parts = str(phase).split()
    if len(parts) >= 3 and parts[1].isdigit():
        return parts[0], int(parts[1]), parts[2]
    return '', None, str(phase)



##### Timed and Recording Wrapper #####

class TimedAgent:
    '''
    Wraps our agent. Records...
        - how long each call takes,
        - a snapshot of the whole board after every phase, so trajectories can be analysed after the game,
        - the learned opponent rates after every movement phase,
        - how well the agent's enemy-move predictions matched what the enemies then ordered.
    Worst-case times are measured here for logging, since the timeout decorator only triggers at the time limit.
    '''

    def __init__(self, factory):
        t0 = time.perf_counter()
        self.agent = factory()
        self.max_time = time.perf_counter() - t0
        self.n_over_limit = int(self.max_time > TIME_LIMIT)

        self.get_actions_ms       = []  # Every get_actions call, in milliseconds
        self.decision_ms_by_phase = {}  # Phase name : get_actions milliseconds
        self.snapshots            = []  # One dict per phase (plus the initial START state)
        self.rate_rows            = []  # Learned opponent rates after each movement phase

        self.prediction_bins = defaultdict(lambda: [0, 0, 0.0, 0.0])  # Predicted chance (tenths) : [predictions, came true, sum of chances, sum of squared errors]
        self._scored_predictions = None  # The enemy_entry_chances_at already scored, so a stale one is never scored twice
        self._last_year = 1901           # Carries the year over to phases that do not name one (e.g. 'COMPLETED')


    def _fn_timed(self, fn, *args):
        '''
        Returns what fn(*args) returns, updating the worst call time and the over-limit count.
        Leaves the call's duration in last_dt.
        '''
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        self.max_time = max(self.max_time, dt)
        self.n_over_limit += int(dt > TIME_LIMIT)
        self.last_dt = dt
        return out


    # The three agent methods called in run_one_game in game.py - timed for StudentAgent
    def new_game(self, game, power_name):
        '''
        Start of a game: times the agent's setup, then snapshots the starting board (not timed).
        '''
        out = self._fn_timed(self.agent.new_game, game, power_name)
        self._snapshot('START', decision_ms=None, observed_movement=False)
        return out


    def update_game(self, all_power_orders):
        '''
        End of a phase: scores the agent's predictions against the orders, times the agent's update, then snapshots the board.
        '''
        game         = self.agent.game
        phase_played = game.phase       # The phase these orders belong to (before process())
        phase_type   = game.phase_type

        if phase_type == 'M':
            self._record_predictions(all_power_orders)  # Before the agent sees the orders
        out = self._fn_timed(self.agent.update_game, all_power_orders)
        self._snapshot(phase_played, self.decision_ms_by_phase.get(phase_played), observed_movement=(phase_type == 'M'))
        return out


    def get_actions(self):
        '''
        Returns the agent's orders for the current phase, recording how long they took.
        '''
        phase = self.agent.game.phase
        out = self._fn_timed(self.agent.get_actions)
        ms = self.last_dt * 1000
        self.get_actions_ms.append(ms)
        self.decision_ms_by_phase[phase] = round(ms, 2)
        return out


    def _snapshot(self, phase_label, decision_ms, observed_movement):
        '''
        Records the board state AFTER the phase was processed: every power's centres and units.
        On a movement phase, also records the learned opponent rates.
        Never lets logging break the game.
        '''
        try:
            game = self.agent.game
            season, year, ptype = parse_phase(phase_label)
            if year is None:
                year = self._last_year
            self._last_year = year
            self.snapshots.append({
                'phase': phase_label, 'year': year, 'season': season, 'phase_type': ptype,
                'centres': {p: len(game.get_centers(p)) for p in ALL_POWERS},
                'units': {p: len(game.get_units(p)) for p in ALL_POWERS},
                'decision_ms': decision_ms,
            })
            if observed_movement:
                self._record_rates(phase_label, year)
        except Exception:
            pass


    def _record_predictions(self, all_power_orders):
        '''
        Scores the agent's enemy-move prediction for this turn against what the enemies then ordered.
        Each (enemy unit, province, chance) the agent predicted is one sample: it came true if that unit
        moved into the province or supported a move into it.
        Never lets logging break the game.
        '''
        try:
            predictions = getattr(self.agent, 'enemy_entry_chances_at', None)
            if not predictions or predictions is self._scored_predictions:
                return
            self._scored_predictions = predictions

            acted_on = {}  # (enemy unit) province : province it moved into, or supported a move into
            for power, orders in all_power_orders.items():
                if power == self.agent.power_name:
                    continue
                for order in orders or []:
                    parts = str(order).split()
                    if len(parts) >= 4 and parts[2] == '-':
                        acted_on[parts[1].split('/')[0]] = parts[3].split('/')[0]
                    elif len(parts) >= 7 and parts[2] == 'S':
                        acted_on[parts[1].split('/')[0]] = parts[6].split('/')[0]

            for province, chances in predictions.items():
                for unit_province, chance in chances:
                    came_true = int(acted_on.get(unit_province) == province)
                    counts = self.prediction_bins[min(int(chance * 10), 9)]
                    counts[0] += 1
                    counts[1] += came_true
                    counts[2] += chance
                    counts[3] += (chance - came_true) ** 2
        except Exception:
            pass


    def _record_rates(self, phase_label, year):
        '''
        Records the agent's learned rates and raw counts for every opponent, after a movement phase.
        '''
        agent = self.agent
        for opp, counts in getattr(agent, 'opponent_counts', {}).items():
            self.rate_rows.append({
                'phase': phase_label, 'year': year, 'opponent_power': opp,
                'defence_rate': round(agent._defence_rate(opp), 4),
                'attack_rate': round(agent._attack_rate(opp), 4),
                'defence_opportunities': counts['defence_opportunities'],
                'defences': counts['defences'],
                'attack_opportunities': counts['attack_opportunities'],
                'attacks': counts['attacks'],
            })



##### Game Setup #####

def build_student(features, debug=False):
    '''
    Returns our agent (current or a frozen version) with this variant's feature flags attached.
    '_agent' in the variant names the module to load; everything else is a flag for the agent.
    '''
    features = dict(features or {})
    module = importlib.import_module(features.pop('_agent', DEFAULT_AGENT))
    agent = module.StudentAgent()
    flags = dict(getattr(agent, 'features', {}))
    flags.update(features)
    agent.features = flags
    agent.debug = debug
    return agent


def build_opponents(scenario, rng):
    '''
    Returns the six opponents for the given scenario, in assignment order:
        - Scenario 1: six static agents,
        - Scenario 2: six agents drawn from S2_POOL,
        - Scenario 3: as Scenario 2, with one of them replaced by the hidden agent stand-in.
    '''
    if scenario == 1:
        return [StaticAgent() for _ in range(6)]

    opponents = [rng.choice(S2_POOL)() for _ in range(6)]
    if scenario == 3:
        opponents[rng.randrange(6)] = HIDDEN_STANDIN()
    return opponents


def outcome_of(centres):
    '''
    Returns the outcome for a final centre count: 'WIN' (18 or more), 'DEFEAT' (none) or 'SURVIVE'.
    '''
    if centres >= WIN_CENTRES:
        return 'WIN'
    if centres == 0:
        return 'DEFEAT'
    return 'SURVIVE'



##### Per-Game Analysis #####

def percentile(values, q):
    '''
    Returns the q-quantile (0 to 1) of the values, taking the nearest lower rank - 0.0 if there are none.
    '''
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def game_stats_from_trajectory(snapshots, power):
    '''
    Returns the derived per-game numbers from the phase snapshots (the player's centres and units over time):
        - peak centres, centres gained and lost, peak and final units,
        - unit_count_drops: rough, drops in unit count (disbands/eliminations), not dislodgements,
        - the first year each centre milestone was reached (blank if never).
    '''
    centres = [s['centres'][power] for s in snapshots]
    units   = [s['units'][power] for s in snapshots]

    gained     = sum(max(b - a, 0) for a, b in zip(centres, centres[1:]))
    lost       = sum(max(a - b, 0) for a, b in zip(centres, centres[1:]))
    units_lost = sum(max(a - b, 0) for a, b in zip(units, units[1:]))

    stats = {
        'peak_centres': max(centres) if centres else 0,
        'centres_gained': gained,
        'centres_lost': lost,
        'final_units': units[-1] if units else 0,
        'peak_units': max(units) if units else 0,
        'unit_count_drops': units_lost,
        'n_phases': max(len(snapshots) - 1, 0),
    }
    for milestone in CENTRE_MILESTONES:
        year = ''
        for snapshot, c in zip(snapshots, centres):
            if c >= milestone:
                year = snapshot['year']
                break
        stats[f'year_to_{milestone}'] = year
    return stats


def trajectory_rows(meta, snapshots, power, agent_names):
    '''
    Returns one row per (snapshot, power): the centres and units of all 7 powers after every phase.
    Our agent's decision time is only filled in on the player's own rows.
    '''
    rows = []
    for index, snapshot in enumerate(snapshots):
        for p in ALL_POWERS:
            rows.append({
                **meta,
                'step': index, 'phase': snapshot['phase'], 'year': snapshot['year'],
                'season': snapshot['season'], 'phase_type': snapshot['phase_type'],
                'power': p, 'agent': agent_names[p], 'is_player': int(p == power),
                'centres': snapshot['centres'][p], 'units': snapshot['units'][p],
                'decision_ms': snapshot['decision_ms'] if p == power else '',
            })
    return rows


def run_scored_game(scenario, repeat, power, features, seed, variant, debug=False):
    '''
    Returns (game row, trajectory rows, opponent-rate rows, prediction bins) for one run_one_game call (game.py:10),
    plus the setup and measurement around it.
    The game row is an ERROR row if anything fails, unless debug=True, which stops on the first failure instead.
    '''
    # The baselines all draw from the shared random module, so seeding it here
    random.seed(seed)
    rng = random.Random(seed)

    meta = {'variant': variant, 'scenario': scenario, 'repeat': repeat, 'power': power, 'seed': seed}
    game_result = dict(meta)
    traj, rates, prediction_bins = [], [], {}

    try:
        player = TimedAgent(lambda: build_student(features, debug))
        opponents = build_opponents(scenario, rng)

        agents      = {}  # power : the agent playing it
        agent_names = {}  # power : the agent's name, 'STUDENT' for ours

        opp_i = 0
        for p in ALL_POWERS:
            if p == power:
                agents[p] = player
                agent_names[p] = 'STUDENT'
            else:
                agents[p] = opponents[opp_i]
                agent_names[p] = opponents[opp_i].agent_name
                opp_i += 1

        t0 = time.perf_counter()
        centres_by_power, year = run_one_game(agents, end_year=END_YEAR)
        game_seconds = time.perf_counter() - t0

        raw_centres = centres_by_power[power]
        centres = min(raw_centres, WIN_CENTRES)
        opp_centres = [c for p, c in centres_by_power.items() if p != power]

        game_result.update({
            'centres': centres,
            'centres_uncapped': raw_centres,
            'outcome': outcome_of(centres),
            'rank': 1 + sum(c > raw_centres for c in opp_centres),  # 1 = most centres (ties share)
            'best_opponent_centres': max(opp_centres) if opp_centres else 0,
            'mean_opponent_centres': round(statistics.fmean(opp_centres), 3) if opp_centres else 0,
            'end_year': year,
            **game_stats_from_trajectory(player.snapshots, power),
            'max_call_time': round(player.max_time, 4),
            'mean_get_actions_ms': round(statistics.fmean(player.get_actions_ms), 2) if player.get_actions_ms else 0,
            'p95_get_actions_ms': round(percentile(player.get_actions_ms, 0.95), 2),
            'calls_over_limit': player.n_over_limit,
            'fallbacks': getattr(player.agent, 'n_fallbacks', ''),
            'game_seconds': round(game_seconds, 2),
            'lineup': '|'.join(f'{p}:{agent_names[p]}' for p in ALL_POWERS),
            'error': '',
        })

        traj = trajectory_rows(meta, player.snapshots, power, agent_names)
        rates = [{**meta, **row, 'opponent_agent': agent_names.get(row['opponent_power'], '')} for row in player.rate_rows]
        prediction_bins = dict(player.prediction_bins)
    except Exception:
        if debug:
            raise
        game_result.update({
            'centres': 0, 'outcome': 'ERROR', 'end_year': '',
            'max_call_time': 0, 'calls_over_limit': 0, 'game_seconds': 0,
            'error': traceback.format_exc(limit=3).replace('\n', ' | '),
        })
    return game_result, traj, rates, prediction_bins



##### Summaries #####

def rubric_points(scenario, win_rate, mean_centres):
    '''
    Returns the points this scenario would score as per the marking rubric.
    '''
    for win_threshold, centre_threshold, points in RUBRIC.get(scenario, []):
        if win_rate > win_threshold or mean_centres > centre_threshold:
            return points
    return 0


def compute_stats(results):
    '''
    Returns a dict of mean supply centres, spread, and outcome rates for a group of games.
    '''
    centres = [result['centres'] for result in results]
    n = len(results)
    return {
        'mean_centres': statistics.fmean(centres),
        'std_centres': statistics.pstdev(centres),
        'win_rate': sum(result['outcome'] == 'WIN' for result in results) / n,
        'survive_rate': sum(result['outcome'] == 'SURVIVE' for result in results) / n,
        'defeat_rate': sum(result['outcome'] == 'DEFEAT' for result in results) / n,
    }


def wilson_interval(wins, n, z=1.96):
    '''
    Returns the (low, high) Wilson score interval for a win rate of wins out of n games (95% by default).
    '''
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def summarise(scenario, game_results):
    '''
    Prints the summary for one scenario - per power, overall, rank, timing, rubric points - and returns the summary dict.
    Returns None if every game errored.
    '''
    completed_results = [result for result in game_results if result['outcome'] != 'ERROR']
    errors = len(game_results) - len(completed_results)
    if not completed_results:
        print(f'Scenario {scenario}: every game errored ({errors} games).')
        print('   ', game_results[0]['error'][:300])
        return None

    results_by_power = defaultdict(list)  # Power (or 'ALL') : its completed game rows
    for result in completed_results:
        results_by_power[result['power']].append(result)
    results_by_power['ALL'] = completed_results

    # Compute all the overall stats and derived numbers for the completed games
    overall          = compute_stats(completed_results)
    n                = len(completed_results)
    wins             = sum(r['outcome'] == 'WIN' for r in completed_results)
    win_lo, win_hi   = wilson_interval(wins, n)
    sem              = overall['std_centres'] / math.sqrt(n)
    max_call_time    = max(result['max_call_time'] for result in completed_results)
    calls_over_limit = sum(result['calls_over_limit'] for result in completed_results)
    mean_ms          = statistics.fmean(r['mean_get_actions_ms'] for r in completed_results)
    p95_ms           = percentile([r['p95_get_actions_ms'] for r in completed_results], 0.95)
    mean_rank        = statistics.fmean(r['rank'] for r in completed_results)
    beat_best        = sum(r['centres_uncapped'] > r['best_opponent_centres'] for r in completed_results) / n
    fallbacks        = [r['fallbacks'] for r in completed_results if r['fallbacks'] != '']
    points           = rubric_points(scenario, overall['win_rate'], overall['mean_centres'])

    print(f'\n----- Scenario {scenario}: Per-Power and Overall Performance of the Player Agent ({n} games{f", {errors} errors" if errors else ""}) -----')

    for power in ALL_POWERS + ['ALL']:
        if not results_by_power[power]:
            continue
        power_stats = compute_stats(results_by_power[power])
        print(f'{power[:3]}: SCs - {power_stats["mean_centres"]:.2f}'
              f'±{power_stats["std_centres"]:.2f}, '
              f'Wins - {power_stats["win_rate"] * 100:.2f}%, '
              f'Survives - {power_stats["survive_rate"] * 100:.2f}%, '
              f'Defeats - {power_stats["defeat_rate"] * 100:.2f}%')

    print(f'Mean SCs {overall["mean_centres"]:.2f} (95% CI ±{1.96 * sem:.2f}) | win rate {overall["win_rate"] * 100:.1f}% (95% Wilson {win_lo * 100:.1f}-{win_hi * 100:.1f}%)')
    print(f'Mean rank among 7 powers: {mean_rank:.2f} | finished ahead of every opponent: {beat_best * 100:.1f}%')
    print(f'Timing: mean decision {mean_ms:.0f} ms, p95 {p95_ms:.0f} ms, worst call {max_call_time * 1000:.1f} ms, calls over the 1 second limit - {calls_over_limit}')

    if fallbacks:
        print(f'Exception fallbacks (empty order lists): {sum(fallbacks)} across {len(fallbacks)} games')
    if calls_over_limit:
        print('*** WARNING: calls exceeded the 1 second limit - the agent would be '
              'disqualified. ***')
    print(f'Rubric: {points}/5 points')
    print('-------------------------------------------------------------------------')

    return {
        'scenario': scenario, 'games': n, 'errors': errors,
        'mean_centres': round(overall['mean_centres'], 3),
        'std_centres': round(overall['std_centres'], 3),
        'ci95_centres': round(1.96 * sem, 3),
        'win_rate': round(overall['win_rate'] * 100, 2),
        'win_ci_low': round(win_lo * 100, 2),
        'win_ci_high': round(win_hi * 100, 2),
        'survive_rate': round(overall['survive_rate'] * 100, 2),
        'defeat_rate': round(overall['defeat_rate'] * 100, 2),
        'mean_rank': round(mean_rank, 3),
        'mean_decision_ms': round(mean_ms, 1),
        'p95_decision_ms': round(p95_ms, 1),
        'max_call_time': round(max_call_time, 4),
        'calls_over_limit': calls_over_limit,
        'fallbacks': sum(fallbacks) if fallbacks else '',
        'points': points,
    }



##### CSV Output #####

def write_csv(path, rows):
    '''
    Writes dict rows to a CSV file, creating its folder if needed.
    Fieldnames are the ordered union of keys, so ERROR rows (which have fewer) cannot break it.
    '''
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)


def append_summary(path, summaries, variant, repeats, base_seed):
    '''
    Appends one row per scenario to results/summary.csv - our running history.
    '''
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    stamp = datetime.now().isoformat(timespec='seconds')
    fields = ['timestamp', 'variant', 'repeats', 'base_seed'] + list(summaries[0].keys())
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for summary in summaries:
            writer.writerow({'timestamp': stamp, 'variant': variant, 'repeats': repeats, 'base_seed': base_seed, **summary})



##### Running #####

def run(scenarios, repeats, features, variant, base_seed, debug=False):
    '''
    Plays every game for one variant (each power, each repeat, each scenario), writes the CSVs and prints the summaries.
    Returns the list of per-game result rows.
    Writes to results/, sharing one timestamp per invocation and variant:
        raw-<variant>-<stamp>.csv    one row per game (final result + derived per-game stats),
        traj-<variant>-<stamp>.csv   one row per (game, phase, power): centres and units of all 7 powers after every phase,
        rates-<variant>-<stamp>.csv  learned opponent-model rates per opponent after every movement phase,
        calib-<variant>-<stamp>.csv  how well the agent's enemy-move predictions matched what the enemies did,
        summary.csv                  running history, one row per (variant, scenario).
    Our agent keeps its own copy of the board, so every power's trajectory (baselines included) is read from it.
    '''
    games = []
    for scenario in scenarios:
        for repeat in range(repeats):
            for idx, power in enumerate(ALL_POWERS):
                # Deterministic, distinct seed per game - identical across variants (paired design)
                seed = base_seed + scenario * 1_000_000 + repeat * 100 + idx
                games.append(dict(scenario=scenario, repeat=repeat, power=power, features=features, seed=seed, variant=variant, debug=debug))

    game_results, traj_rows, rate_rows = [], [], []
    prediction_totals = defaultdict(lambda: [0, 0, 0.0, 0.0])  # (scenario, chance bin) : counts summed over its games
    for game in tqdm(games, desc=variant):
        row, traj, rates, prediction_bins = run_scored_game(**game)
        game_results.append(row)
        traj_rows.extend(traj)
        rate_rows.extend(rates)
        for chance_bin, counts in prediction_bins.items():
            totals = prediction_totals[(game['scenario'], chance_bin)]
            for i, count in enumerate(counts):
                totals[i] += count

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    raw_path   = os.path.join(RESULTS_DIR, f'raw-{variant}-{stamp}.csv')
    traj_path  = os.path.join(RESULTS_DIR, f'traj-{variant}-{stamp}.csv')
    rates_path = os.path.join(RESULTS_DIR, f'rates-{variant}-{stamp}.csv')

    write_csv(raw_path, game_results)
    write_csv(traj_path, traj_rows)
    write_csv(rates_path, rate_rows)

    if prediction_totals:
        # How well the agent's enemy-move prediction matched what the enemies then did (new technique 3)
        write_csv(os.path.join(RESULTS_DIR, f'calib-{variant}-{stamp}.csv'), [
            {'variant': variant, 'scenario': scenario, 'chance_from': chance_bin / 10, 'predictions': counts[0], 'came_true': counts[1], 'sum_chance': round(counts[2], 4), 'sum_squared_error': round(counts[3], 4)}
            for (scenario, chance_bin), counts in sorted(prediction_totals.items())
        ])

    summaries = []
    for scenario in scenarios:
        scenario_results = [r for r in game_results if r['scenario'] == scenario]
        summary = summarise(scenario, scenario_results)
        if summary:
            summaries.append(summary)
    if summaries:
        append_summary(os.path.join(RESULTS_DIR, 'summary.csv'), summaries, variant, repeats, base_seed)

    total = sum(summary['points'] for summary in summaries)
    print(f'\nVariant "{variant}": {total}/{5 * len(summaries)} agent points across scenarios {scenarios}')
    print(f'Per-game results: {raw_path}\nTrajectories:     {traj_path}\nOpponent rates:   {rates_path}')
    return game_results



##### Paired Comparison of Two Runs #####

def load_raw(path):
    '''
    Returns (scenario, repeat, power, seed) : game row, for every game in a raw-*.csv that did not error.
    '''
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['outcome'] != 'ERROR']
    return {(r['scenario'], r['repeat'], r['power'], r['seed']): r for r in rows}


def paired_ci(diffs):
    '''
    Returns (mean, 95% half-width) of the paired differences - half-width 0 for fewer than two.
    '''
    n = len(diffs)
    mean = statistics.fmean(diffs)
    if n < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(diffs) / math.sqrt(n)


def compare_runs(path_a, path_b):
    '''
    Paired comparison of two raw-*.csv files over the same (scenario, repeat, power, seed) games, B minus A.
    Reports, per scenario:
        - the mean difference in centres, win rate and rank with 95% CIs,
        - how many games B was better, equal or worse in,
        - the difference in mean decision time.
    '''
    a, b = load_raw(path_a), load_raw(path_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        print('No shared games between the two files (different seeds or scenarios?).')
        return
    print(f'A = {path_a}\nB = {path_b}\nShared games: {len(shared)} (B - A; positive means B is better)\n')

    for scenario in sorted({key[0] for key in shared}):
        keys = [k for k in shared if k[0] == scenario]

        d_centres = [float(b[k]['centres']) - float(a[k]['centres']) for k in keys]
        d_win = [int(b[k]['outcome'] == 'WIN') - int(a[k]['outcome'] == 'WIN') for k in keys]
        d_rank = [float(a[k]['rank']) - float(b[k]['rank']) for k in keys]  # Positive = B ranks better
        d_ms = [float(b[k]['mean_get_actions_ms']) - float(a[k]['mean_get_actions_ms']) for k in keys]

        better = sum(x > 0 for x in d_centres)
        worse = sum(x < 0 for x in d_centres)

        mc, cc = paired_ci(d_centres)
        mw, cw = paired_ci(d_win)
        mr, cr = paired_ci(d_rank)

        print(f'Scenario {scenario} (n={len(keys)}): '
              f'dCentres {mc:+.2f} ±{cc:.2f} | dWin rate {mw * 100:+.1f}pp ±{cw * 100:.1f} | '
              f'dRank (positive = B better) {mr:+.2f} ±{cr:.2f} | '
              f'B better/equal/worse in {better}/{len(keys) - better - worse}/{worse} games | '
              f'dDecision time {statistics.fmean(d_ms):+.0f} ms')
        print('   an interval that excludes 0 suggests a real difference at ~95%\n')



##### Entry Point #####

def main():
    '''
    Parses the command line, then runs each chosen variant (or the paired comparison, if asked for).
    '''
    # The baselines turn sets of strings into lists before picking from them, so their choices
    # follow the hash seed rather than random.seed - without this, scenarios 2 and 3 do not repeat
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, *sys.argv])

    parser = argparse.ArgumentParser(description='CITS3011 Group 30 experiments')
    parser.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument('--repeats', type=int, default=10, help='games per power per scenario (7 powers per repeat)')
    parser.add_argument('--variant', nargs='+', default=['default'], help=f'one or more of {sorted(VARIANTS)}, "all" or "frozen"')
    parser.add_argument('--seed', type=int, default=3011, help='base random seed')
    parser.add_argument('--debug-agent', action='store_true', help='let agent exceptions surface instead of being caught')
    parser.add_argument('--quick', action='store_true', help='one game per power (smoke test)')
    parser.add_argument('--compare', nargs=2, metavar=('RAW_A', 'RAW_B'), help='paired comparison of two raw-*.csv files, then exit')
    args = parser.parse_args()

    if args.compare:
        compare_runs(*args.compare)
        return

    if args.variant == ['all']:
        variants = sorted(VARIANTS)
    elif args.variant == ['frozen']:
        # Plain frozen versions only - no extra flags alongside '_agent'
        variants = sorted(name for name, flags in VARIANTS.items() if list(flags) == ['_agent'])
    else:
        variants = args.variant

    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f'unknown variant(s) {unknown}; choose from {sorted(VARIANTS)}')

    repeats = 1 if args.quick else args.repeats
    for variant in variants:
        print(f'Variant "{variant}" | scenarios {args.scenarios} | {repeats * 7 * len(args.scenarios)} games | seed {args.seed}')
        run(scenarios=args.scenarios, repeats=repeats, features=VARIANTS[variant], variant=variant, base_seed=args.seed, debug=args.debug_agent)


if __name__ == '__main__':
    main()
