https://matsuoinstitute.github.io/cabt/




# Differences Between the Official Pokémon TCG Rules and the Simulator Behavior
The simulator used in this competition is designed for AI-vs-AI battles, and its rules and behavior may differ in some respects from those of the official Pokémon Trading Card Game. Below is a summary of the differences we are currently aware of.

Some attacks may not be selectable in the simulator even when they could be declared under the official rules In the official Pokémon TCG, there are cases where a player is allowed to declare an attack, but the effect cannot be fully resolved, and the turn simply ends after the attack declaration. In the simulator, such attacks may instead be treated as not selectable from the beginning. Examples include the following cases: Using an attack with an effect that puts a Basic Pokémon from the deck onto the Bench when there is no open Bench space Using an attack with an effect that draws cards when the player’s deck has 0 cards remaining Using an attack with an effect that interacts with the opponent’s hand when the opponent has 0 cards in hand Although the handling is different, we believe the end result is the same, and the impact on gameplay is minimal.

About Nullifying Zero, the attack of Mega Zygarde ex For Mega Zygarde ex’s attack, Nullifying Zero, under the official Pokémon TCG rules, the player using the attack may choose the order in which damage is assigned to the targets. In the simulator, however, the target order cannot be chosen, and coins are flipped automatically from left to right. This differs from the official rules, but since Knock Out processing is handled simultaneously, we believe this does not affect the competition.

Prize-taking order when both players’ Pokémon are Knocked Out at the same time When both players’ Pokémon are Knocked Out at the same time, the order of taking Prize cards differs between the official Pokémon TCG rules and the simulator.

Official Pokémon TCG order

The player whose turn is next chooses their Prize cards
The opposing player chooses their Prize cards
Both players take their Prize cards at the same time
The player whose turn is next puts a Pokémon into the Active Spot first
Simulator order used in this competition

The player whose turn is next chooses their Prize cards
That player takes their Prize cards
The opposing player chooses their Prize cards
The opposing player takes their Prize cards
The player whose turn is next puts a Pokémon into the Active Spot first
This is a different processing order from the official rules. However, in this competition, even if both players ultimately take all of their Prize cards, the result is treated as a draw, so we believe this does not affect match outcomes.

In this competition, please note that the simulator behavior will be treated as the correct behavior. If we identify any additional points that should be announced, we will share them in the Discussion forum as needed.



# BASELINE EXPERIMENT

----------------------------------------
ID : 73467baa-8be8-11f1-9ba0-201e8805b73d
Name : cabt
Title : Card Battle
Description : Limited Card Battle.
Version : 1.0.0
Module Version : 1.32.2
Configuration : {'episodeSteps': 10000000, 'actTimeout': 0, 'runTimeout': 2000}
Specification : {'action': {'description': 'List of option index.', 'type': 'array', 'default': []}, 'agents': [2], 'configuration': {'episodeSteps': {'description': 'Maximum number of steps in the episode.', 'type': 'integer', 'minimum': 1, 'default': 10000000}, 'actTimeout': {'description': 'Maximum runtime (seconds) to obtain an action from an agent.', 'type': 'number', 'minimum': 0, 'default': 0}, 'runTimeout': {'description': 'Maximum runtime (seconds) of an episode (not necessarily DONE).', 'type': 'number', 'minimum': 0, 'default': 2000}}, 'info': {}, 'observation': {'remainingOverageTime': {'description': 'Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0.', 'shared': False, 'type': 'number', 'minimum': 0, 'default': 600}, 'step': {'description': 'Current step within the episode.', 'type': 'integer', 'shared': True, 'minimum': 0, 'default': 0}}, 'reward': {'description': 'Lost:-1, Won:1, Draw:0', 'enum': [-1, 0, 1], 'default': 0, 'type': ['number', 'null']}}
Steps : [...] // Left for brevity.
Rewards : [1, -1]
Statuses : ['DONE', 'DONE']
Schema Version : 1
Info : {}
----------------------------------------