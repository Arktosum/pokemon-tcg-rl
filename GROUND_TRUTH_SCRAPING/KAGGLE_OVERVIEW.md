# The Pokémon Company - PTCG AI Battle Challenge Simulation
Build an AI Training Agent to play the Pokémon Trading Card Game


# Overview
This project aims to enhance the performance of an AI Training Agent with the Pokémon Trading Card Game (TCG). The research focuses on training AI Training Agents for competitive play in a system where probability, unknown elements, and strategic planning are key determinants of success.


# Description
Note: This TCG AI Battle Challenge has two competitions. This competition is the Simulation competition. Learn more about the Hackathon here. Participation in the Hackathon is not required to enter this competition.

As a strategic game of its own, Pokémon TCG players must make gameplay decisions while being mindful of their opponent's own strategies, decks and hands. When formulating their approach to the game, players need to take into consideration various Pokémon types and thousands of different card combinations to consider various gameplay possibilities. In addition, other factors such as card draws and coin tosses introduce additional gameplay variables. The AI Training Agent’s training will be conducted within this competitive simulation framework.

Not knowing what cards an opponent holds presents a core challenge for an AI Training Agent. Participants are encouraged to explore novel methodologies for strategy learning and decision making in this dynamic environment.

Participants will be provided with a simulator (SDK) for training and testing. This toolkit uses the same logic as the Kaggle competition environment, making it suitable for local debugging and reinforcement learning.

Using rule-based programming alone may not ensure a high ranking. Winning a Pokémon TCG game requires forward thinking, real-time adaptation, and optimal decision-making. With many different deck variations available and evolving strategies, no two games are alike. The AI Training Agent must demonstrate high analytical capacity, adaptability, and be ready for the unexpected.

# Evaluation
Each day your team is able to submit up to 5 agents to the competition. Each submission will play Episodes (games) against other agents on the ladder that have a similar skill rating. Over time skill ratings will go up with wins or down with losses and evened out with ties. To reduce the number of agents playing and increase the number of episodes each team participates in, we only track the latest 2 submissions and use those for final submissions.

Every agent submitted will continue to play episodes until the end of the competition, with newer agents playing a much more frequent number of episodes. On the leaderboard only your best scoring agent will be shown, but you can track the progress of all of your submissions on your Submissions page.

Each Submission has an estimated Skill Rating which is modeled by a Gaussian N(μ,σ2) where μ is the estimated skill and σ represents the uncertainty of that estimate which will decrease over time.

When you upload a Submission, we first play a Validation Episode where that Submission plays against copies of itself to make sure it works properly. If the Episode fails, the Submission is marked as Error and you can download the agent logs to help figure out why. Otherwise, we initialize the Submission with μ0=600 and it joins the pool of All Submissions for ongoing evaluation.

We repeatedly run Episodes from the pool of All Submissions, and try to pick Submissions with similar ratings for fair matches. Newly submitted agents will be given an increased rate in the number of episodes run to give you faster feedback.

# Ranking System

After an Episode finishes, we'll update the Rating estimate for all Submissions in that Episode. If one Submission won, we'll increase its μ and decrease its opponent's μ -- if the result was a draw, then we'll move the two μ values closer towards their mean. The updates will have magnitude relative to the deviation from the expected result based on the previous μ values, and also relative to each Submission's uncertainty σ. We also reduce the σ terms relative to the amount of information gained by the result. The score by which your agent wins or loses an Episode does not affect the skill rating updates.

Final Evaluation At the submission deadline on August 16, 2026, additional submissions will be locked. From August 16, 2026 for approximately two weeks, we will continue to run games. At the conclusion of this period, the leaderboard is final.


# How to Play Pokémon TCG
Download the most recent rulebook for the Pokémon Trading Card Game.

View the Competition Data Page for more information on cards and decks available for this tournament.

Get information on the Pokémon TCG, the Play! Pokémon program and more on The Pokémon Company's Rules & Resources page.

Simulator API Documentation
Battles are run on the cabt Engine, a Pokémon TCG battle simulator built for kaggle-environments.

Each turn, your agent receives an observation — including game logs, the current board state, and a list of legal options — and returns the indices of the options it selects. The engine only ever presents legal moves.

API documentation: https://matsuoinstitute.github.io/cabt/

Note, there are a few differences between the official Pokémon TCG rules and the simulator behavior, which can be found here.



# How to Submit to this Competition

Submissions need to be a .tar.gz bundle with main.py at the top level directory (not nested) and include a deck.csv. To create a submission, create the .tar.gz with tar -czvf submission.tar.gz *. Upload this under the My Submissions tab and you should be good to go! Your submission will start with a scheduled game vs itself to ensure everything is working before being entered into the matchmaking pool against the rest of the leaderboard.

Code and configuration for The TCG AI Battle Challenge as of kaggle-environments 1.14.10. See the latest at https://github.com/Kaggle/kaggle-environments.


# Frequently Asked Questions
## Submissions

Submissions must be at most 197.7 MiB
Daily submission limit 5
Only your most recent 2 are active
Your files will be located in /kaggle_simulations/agent/. Ensure all your file imports are set appropriately
## Submission Resources
HDD Space: 11.8 GiB
RAM: 12.2 GiB
vCPUs: 2
Submission Size Limit: 197.7 MiB



# Kaggle Dockefile


# This is set in cloudbuild.yaml, but provide a default
ARG BASE_IMAGE=gcr.io/kaggle-images/python:v163

# ===== Base Stages =====
FROM node:22-slim AS node_builder
FROM ${BASE_IMAGE} AS base

COPY --from=node_builder /usr/local/ /usr/local/
# Confirm the installation was successful
RUN node -v && npm -v

WORKDIR /usr/src/app/kaggle_environments

# Copy everything required for the build.
# This includes the dependency list, the source code, and manifest.
COPY ./pyproject.toml ./pyproject.toml
COPY ./MANIFEST.in ./MANIFEST.in
COPY ./kaggle_environments ./kaggle_environments

# Now that the source code is present, run the installation.
# This layer is now correctly cached and will only rebuild if the source or pyproject.toml changes.
RUN uv pip install --system . && \
    uv cache clean

# 3. Copy any remaining files that aren't part of the build.
COPY ./README.md ./README.md

# ===== CPU Final Stage =====
FROM base AS cpu
CMD ["kaggle-environments"]

# ===== GPU Final Stage =====
FROM base AS gpu
CMD ["kaggle-environments"]



# Reminder about the Kaggle Simulation Competition Format
Hi everyone,

We have a number of participants who are new to Kaggle competitions and/or simulations and have raised some questions about the format of this competition, how games are run, and how the winners are determined. Here's a brief review.

You are allowed to have two active submissions at any time on the leaderboard and your best performing submission will be your leaderboard score. You can submit as often as you'd like, up to 5 times per day, and your two most recent submissions will be your active ones.
Your submissions will continually play games against other submissions throughout the duration of the competition, as well as for approximately two weeks beyond the competition deadline. Your leaderboard ranking will continue to shift throughout the competition, and we run games after the competition to:
Ensure that new submissions have time to calibrate to their appropriate ranking on the leaderboard
Play enough games to increase the confidence (decrease σ) such that the leaderboard positions have converged.
Every game played meaningfully updates your ranking - unfavorable or favorable matchmaking early on in a submission's lifecycle is never "locked in."
The two submissions you have active by the end of the submission timeline will be the ones used in the final leaderboard calibration/convergence period. You're encouraged to test different strategies throughout the competition, but its those two that will be tested in the end, and after enough games are run, will determine the final leaderboard.
We do not intend to reset the leaderboard scores at the deadline for a final tournament, but reserve the right to do in conjunction with the Sponsor team. We'd announce any changes to the format with plenty of notice and would only do so if deemed necessary.
We aim to have each submission plan dozens of games per day (currently targeting 24/day) with prioritization given to newer submissions and submissions that haven't run in a while. That said, you may see long delays between some episode runs - that's not out of the ordinary. With so many participants and submissions made each day, it's likely that you may encounter some matchmaking lag, but that will resolve itself within the day.
We're so excited to see how the competition is continuing to unfold!

Kaggle Team
