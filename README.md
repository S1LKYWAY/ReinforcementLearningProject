# ReinforcementLearningProject

# README for Reinforcement Learning Project

## Overview
This project focuses on implementing value iteration and Q-learning. The agents developed in this project are initially tested on Gridworld and later applied to a simulated robot controller (Crawler) and Pacman. The project includes an autograder for self-assessment.
Project completed for SFU CMPT 310, adapted from UC Berkeley CS 188

## Project Structure
The primary files edited:
- **valueIterationAgents.py:** Implements a value iteration agent for solving known MDPs.
- **qlearningAgents.py:** Contains Q-learning agents for Gridworld, Crawler, and Pacman.
- **analysis.py:** A file to record answers to questions posed in the project.

### Files Read but Not Edited:
- **mdp.py:** Defines methods on general MDPs.
- **learningAgents.py:** Defines the base classes `ValueEstimationAgent` and `QLearningAgent`, which agents extend.
- **util.py:** Provides utilities, including `util.Counter`, particularly useful for Q-learners.
- **gridworld.py:** The Gridworld implementation.
- **featureExtractors.py:** Classes for extracting features on (state, action) pairs. Used for the approximate Q-learning agent (in `qlearningAgents.py`).

### Supporting Files (Ignored):
- **environment.py:** Abstract class for general reinforcement learning environments. Used by `gridworld.py`.
- **graphicsGridworldDisplay.py:** Gridworld graphical display.
- **graphicsUtils.py:** Graphics utilities.
- **textGridworldDisplay.py:** Plug-in for the Gridworld text interface.
- **crawler.py:** The crawler code and test harness. Run this but do not edit it.
- **graphicsCrawlerDisplay.py:** GUI for the crawler robot.
- **autograder.py:** Project autograder
- **testParser.py:** Parses autograder test and solution files
- **testClasses.py:** General autograding test classes
- **test_cases/:** Directory containing the test cases for each question
- **reinforcementTestClasses.py:** Project 3 specific autograding test classes.

## Contributers
- Qasim Abbas
