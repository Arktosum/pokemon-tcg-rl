# 00_GROUND_TRUTH

## 0. GROUND-TRUTH ANCHORING (RAW TEXT)
**Source URLs:** 
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data

### KAGGLE OVERVIEW TEXT (FULL RAW SCRAPE)
menu
Skip to
content
Create
explore

Home

emoji_events

Competitions

leaderboard

Benchmarks

smart_toy

Game Arena

code

Data Hub

expand_more
format_list_bulleted

More

expand_more
search
​
Sign In
Register
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
THE POKÉMON COMPANY - PTCGABC TEAM · FEATURED SIMULATION COMPETITION · 22 DAYS TO GO
Join Competition
more_horiz
The Pokémon Company - PTCG AI Battle Challenge Simulation

Build an AI Training Agent to play the Pokémon Trading Card Game

Overview
Data
Code
Models
Discussion
Leaderboard
Rules
Overview

This project aims to enhance the performance of an AI Training Agent with the Pokémon Trading Card Game (TCG). The research focuses on training AI Training Agents for competitive play in a system where probability, unknown elements, and strategic planning are key determinants of success.

Start

a month ago

Close

22 days to go
Merger & Entry
Description
link
keyboard_arrow_up

Note: This TCG AI Battle Challenge has two competitions. This competition is the Simulation competition. Learn more about the Hackathon here. Participation in the Hackathon is not required to enter this competition.

As a strategic game of its own, Pokémon TCG players must make gameplay decisions while being mindful of their opponent's own strategies, decks and hands. When formulating their approach to the game, players need to take into consideration various Pokémon types and thousands of different card combinations to consider various gameplay possibilities. In addition, other factors such as card draws and coin tosses introduce additional gameplay variables. The AI Training Agent’s training will be conducted within this competitive simulation framework.

Not knowing what cards an opponent holds presents a core challenge for an AI Training Agent. Participants are encouraged to explore novel methodologies for strategy learning and decision making in this dynamic environment.

Participants will be provided with a simulator (SDK) for training and testing. This toolkit uses the same logic as the Kaggle competition environment, making it suitable for local debugging and reinforcement learning.

Using rule-based programming alone may not ensure a high ranking. Winning a Pokémon TCG game requires forward thinking, real-time adaptation, and optimal decision-making. With many different deck variations available and evolving strategies, no two games are alike. The AI Training Agent must demonstrate high analytical capacity, adaptability, and be ready for the unexpected.

Evaluation
link
keyboard_arrow_up

Each day your team is able to submit up to 5 agents to the competition. Each submission will play Episodes (games) against other agents on the ladder that have a similar skill rating. Over time skill ratings will go up with wins or down with losses and evened out with ties. To reduce the number of agents playing and increase the number of episodes each team participates in, we only track the latest 2 submissions and use those for final submissions.

Every agent submitted will continue to play episodes until the end of the competition, with newer agents playing a much more frequent number of episodes. On the leaderboard only your best scoring agent will be shown, but you can track the progress of all of your submissions on your Submissions page.

Each Submission has an estimated Skill Rating which is modeled by a Gaussian N(μ,σ2) where μ is the estimated skill and σ represents the uncertainty of that estimate which will decrease over time.

When you upload a Submission, we first play a Validation Episode where that Submission plays against copies of itself to make sure it works properly. If the Episode fails, the Submission is marked as Error and you can download the agent logs to help figure out why. Otherwise, we initialize the Submission with μ0=600 and it joins the pool of All Submissions for ongoing evaluation.

We repeatedly run Episodes from the pool of All Submissions, and try to pick Submissions with similar ratings for fair matches. Newly submitted agents will be given an increased rate in the number of episodes run to give you faster feedback.

Ranking System

After an Episode finishes, we'll update the Rating estimate for all Submissions in that Episode. If one Submission won, we'll increase its μ and decrease its opponent's μ -- if the result was a draw, then we'll move the two μ values closer towards their mean. The updates will have magnitude relative to the deviation from the expected result based on the previous μ values, and also relative to each Submission's uncertainty σ. We also reduce the σ terms relative to the amount of information gained by the result. The score by which your agent wins or loses an Episode does not affect the skill rating updates.

Final Evaluation
At the submission deadline on August 16, 2026, additional submissions will be locked. From August 16, 2026 for approximately two weeks, we will continue to run games. At the conclusion of this period, the leaderboard is final.

Timeline
link
keyboard_arrow_up

June 16, 2026 11:00 am UTC - Start Date.

August 9, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.

August 9, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.

August 16, 2026 - Final Submission Deadline.

August 17, 2026 to (approx.) August 31, 2026 - We will continue to run games, or until the leaderboard has reached convergence. At the conclusion of this period, the leaderboard is final.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Prizes
link
keyboard_arrow_up

The Competition track itself does not include monetary prizes. However, participants who submit a report to the Hackathon track will be eligible for prize awards. Final rankings for Hackathon prizes will be determined based on both the Competition leaderboard performance and the Hackathon evaluation.

How to Play Pokémon TCG
link
keyboard_arrow_up

Download the most recent rulebook for the Pokémon Trading Card Game.

View the Competition Data Page for more information on cards and decks available for this tournament.

Get information on the Pokémon TCG, the Play! Pokémon program and more on The Pokémon Company's Rules & Resources page.

Simulator API Documentation

Battles are run on the cabt Engine, a Pokémon TCG battle simulator built for kaggle-environments.

Each turn, your agent receives an observation — including game logs, the current board state, and a list of legal options — and returns the indices of the options it selects. The engine only ever presents legal moves.

API documentation: https://matsuoinstitute.github.io/cabt/

Note, there are a few differences between the official Pokémon TCG rules and the simulator behavior, which can be found here.

How to Submit to this Competition
link
keyboard_arrow_up

Submissions need to be a .tar.gz bundle with main.py at the top level directory (not nested) and include a deck.csv. To create a submission, create the .tar.gz with tar -czvf submission.tar.gz *. Upload this under the My Submissions tab and you should be good to go! Your submission will start with a scheduled game vs itself to ensure everything is working before being entered into the matchmaking pool against the rest of the leaderboard.

Code and configuration for The TCG AI Battle Challenge as of kaggle-environments 1.14.10. See the latest at https://github.com/Kaggle/kaggle-environments.

Trademark Note
link
keyboard_arrow_up

©Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo

Frequently Asked Questions
link
keyboard_arrow_up
Submissions
Submissions must be at most 197.7 MiB
Daily submission limit 5
Only your most recent 2 are active
Your files will be located in /kaggle_simulations/agent/. Ensure all your file imports are set appropriately
Submission Resources
HDD Space: 11.8 GiB
RAM: 12.2 GiB
vCPUs: 2
Submission Size Limit: 197.7 MiB

For questions about the environment OS or python env, please see:

Docker Image
Citation
link
keyboard_arrow_up

The Pokémon Company, HEROZ, Matsuo Institute, Addison Howard, Bovard Doerschuk-Tiberi. The Pokémon Company - PTCG AI Battle Challenge Simulation. https://kaggle.com/competitions/pokemon-tcg-ai-battle, 2026. Kaggle.

Cite
Competition Host

The Pokémon Company - PTCGABC Team

Prizes & Awards

Knowledge

Awards Points & Medals

Participation

12,415 Entrants

6,583 Participants

5,735 Teams

10,836 Submissions

Tags
Games
Artificial Intelligence
Reinforcement Learning
Custom Metric
Table of Contents
collapse_all
Overview
Description
Evaluation
Timeline
Prizes
How to Play Pokémon TCG
How to Submit to this Competition
Trademark Note
Frequently Asked Questions
Citation

### KAGGLE DATA TEXT (FULL RAW SCRAPE)
menu
Skip to
content
Create
explore

Home

emoji_events

Competitions

leaderboard

Benchmarks

smart_toy

Game Arena

code

Data Hub

expand_more
format_list_bulleted

More

expand_more
search
​
Sign In
Register
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
THE POKÉMON COMPANY - PTCGABC TEAM · FEATURED SIMULATION COMPETITION · 22 DAYS TO GO
Join Competition
more_horiz
The Pokémon Company - PTCG AI Battle Challenge Simulation

Build an AI Training Agent to play the Pokémon Trading Card Game

Overview
Data
Code
Models
Discussion
Leaderboard
Rules
Dataset Description
Episode Replay

You can access episode replays for your Submissions from the Submissions tab, or you can download them via the CLI (or MCP) read more here: https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md

You can download replay files from other teams from the Leaderboard, and we will enable a daily episode export of the top rated episodes (to help BC/RL/IL). This will be posted in the competition forums.

Dataset Description

This competition provides card metadata and reference materials for the Pokémon Trading Card Game environment used in the simulator.

The dataset contains card identifiers, card names, expansion information, and reference images that correspond to the cards available in the competition environment. These files are provided to help participants understand the card pool and map card IDs used in the simulator to their corresponding card details.

Two versions of the card data are provided: an English version and a Japanese version. The content is identical except for the language of the card names and descriptions.

Files

The dataset includes the following files:

Card\_ID\_List\_EN.pdf - A reference document listing all cards available in the competition environment. Each entry includes the card ID, card name, expansion set, collection number, and an image of the card.

Card\_ID\_List\_JP.pdf - The Japanese version of the card reference document. The structure is identical to the English version, but card names and descriptions are written in Japanese.

EN Card Data.csv - A structured dataset containing metadata for each card in English.

JP Card Data.csv - The Japanese version of the card metadata dataset.

CSV File Structure

Both CSV files share the same schema and contain structured metadata for all cards used in the competition environment.

Card ID
Unique identifier for each card used by the simulator.

Card Name
The name of the card.

Expansion
The expansion set the card belongs to.

Collection No.
The card's collection number within the expansion.

Stage (Pokémon) / Type (Energy and Trainer)
Indicates the Pokémon evolution stage (Basic, Stage 1, Stage 2) or the card type for Energy and Trainer cards.

Rule
Special rule text associated with the card, if applicable.

Category
The category of the card (e.g., Pokémon, Trainer, Energy).

Previous stage
The previous evolution stage required for this Pokémon card.

HP
Hit Points of the Pokémon.

Type
Pokémon type (e.g., Grass, Fire, Water).

Weakness
Type weakness of the Pokémon.

Resistance (Type)
Type resistance of the Pokémon.

Retreat
Retreat cost required to switch the Pokémon.

Move Name
Name of the attack or move.

Cost
Energy cost required to use the move.

Damage
Damage dealt by the move.

Effect Explanation
Description of the move effect or additional rule text.

Files

60 files

Size

327.59 MB

Type

h, py, csv + 11 others

License

Subject to Competition Rules

Card_ID List_EN.pdf(137.65 MB)
get_app
fullscreen
chevron_right

Competition Rules

To see this data you need to agree to the competition rules.
Please sign in or register to accept the rules.

Sign In
Data Explorer

327.59 MB

arrow_right
folder

ptcg_engine

arrow_right
folder

sample_submission

drive_pdf

Card_ID List_EN.pdf

drive_pdf

Card_ID List_JP.pdf

calendar_view_week

EN_Card_Data.csv

calendar_view_week

JP_Card_Data.csv

Summary
arrow_right
folder

60 files

arrow_right
calendar_view_week

35 columns

get_app
Download All
text_snippet
Metadata
License
Subject to Competition Rules
