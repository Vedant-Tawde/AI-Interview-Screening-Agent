# AI Interview Screening Agent

This project implements an OpenEnv-compatible environment for simulating a multi-stage AI hiring pipeline.

## Environment Description
The environment tracks a candidate's journey through four distinct stages:
1. **Screening**: Evaluation of resume vs job description.
2. **Interview**: Interactive Q&A session.
3. **Evaluation**: Scored assessment of candidate responses.
4. **Decision**: Final hiring or rejection decision.

## Action Space
- `shortlist_candidate`: Transitions the agent to the interview stage.
- `reject_candidate`: Terminates the episode with a final decision.
- `ask_question(question: str)`: Submits a question to the candidate simulator.
- `evaluate_answer(score: float)`: Provides a numerical score for the previous answer.
- `hire_candidate`: Final successful outcome.
- `reject_after_interview`: Final negative outcome.

## Reward Design
- **Partial Rewards**: Encourages correct progression (e.g., +0.2 for correct screening).
- **Technical Accuracy**: Rewards relevant questions (+0.1) and accurate evaluations (+0.3).
- **Goal Achievement**: Full success (+1.0) granted for correct final hiring decisions.
- **Penalties**: Deduction for incorrect decisions or irrelevant interactions.

## Task Descriptions
- **Task 1 (Easy)**: Focuses on basic resume matching.
- **Task 2 (Medium)**: Requires managing the interview loop and scoring.
- **Task 3 (Hard)**: Full end-to-end pipeline execution.

## Setup & Execution
### Running with Docker
```bash
docker build -t ai-hiring-agent .
docker run -e API_BASE_URL="xxx" -e MODEL_NAME="xxx" -e HF_TOKEN="xxx" ai-hiring-agent
```

### Direct Execution
```bash
pip install pydantic openai pyyaml
python inference.py
```

## Expected Baseline Behavior
The agent should identify skills in the JOB DESCRIPTION and ask relevant technical questions. It must then evaluate the candidate's response (simulated based on their skill set) and decide whether to hire them based on the cumulative scores.
