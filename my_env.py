import pydantic
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json

class Stage(str, Enum):
    SCREENING = "screening"
    INTERVIEW = "interview"
    EVALUATION = "evaluation"
    DECISION = "decision"

class CandidateProfile(pydantic.BaseModel):
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]

class JobDescription(pydantic.BaseModel):
    title: str
    required_skills: List[str]
    min_experience: int

class ActionType(str, Enum):
    SHORTLIST = "shortlist_candidate"
    REJECT_SCREENING = "reject_candidate"
    ASK_QUESTION = "ask_question"
    EVALUATE_ANSWER = "evaluate_answer"
    HIRE = "hire_candidate"
    REJECT_INTERVIEW = "reject_after_interview"

class Action(pydantic.BaseModel):
    action_type: ActionType
    question: Optional[str] = None
    score: Optional[float] = None

class Observation(pydantic.BaseModel):
    stage: Stage
    candidate: Optional[CandidateProfile] = None
    job: JobDescription
    last_answer: Optional[str] = None
    history: List[Dict[str, str]] = []
    done: bool = False

class Reward(pydantic.BaseModel):
    value: float
    reason: str

class InterviewEnv:
    def __init__(self, config: Dict[str, Any]):
        self.candidate = CandidateProfile(**config["candidate"])
        self.job = JobDescription(**config["job"])
        self.target_questions = config.get("target_questions", 3)
        self.reset()

    def reset(self) -> Observation:
        self.current_stage = Stage.SCREENING
        self.history = []
        self.questions_asked = 0
        self.evaluations = []
        self.done = False
        return self.state()

    def state(self) -> Observation:
        return Observation(
            stage=self.current_stage,
            candidate=self.candidate if self.current_stage == Stage.SCREENING else None,
            job=self.job,
            last_answer=self.history[-1]["content"] if self.history and self.history[-1]["role"] == "candidate" else None,
            history=self.history,
            done=self.done
        )

    def _simulate_answer(self, question: str) -> str:
        # Simple deterministic simulation
        relevance = sum(1 for skill in self.job.required_skills if skill.lower() in question.lower())
        skill_match = sum(1 for skill in self.candidate.skills if skill.lower() in question.lower())
        
        if skill_match > 0:
            return f"I have extensive experience with relevant technologies. In my projects like {', '.join(self.candidate.projects[:2])}, I applied these skills effectively."
        elif relevance > 0:
            return "I am familiar with the concepts, though I haven't used this specific tool in a production environment yet."
        else:
            return "That's an interesting question. I'd need to research that specific area more deeply to provide a comprehensive answer."

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict]:
        reward_val = 0.0
        reason = ""

        if self.current_stage == Stage.SCREENING:
            if action.action_type == ActionType.SHORTLIST:
                met_reqs = all(s in self.candidate.skills for s in self.job.required_skills) and \
                           self.candidate.experience_years >= self.job.min_experience
                if met_reqs:
                    reward_val = 0.2
                    reason = "Correctly shortlisted qualified candidate"
                    self.current_stage = Stage.INTERVIEW
                else:
                    reward_val = -0.2
                    reason = "Shortlisted unqualified candidate"
                    self.done = True
            elif action.action_type == ActionType.REJECT_SCREENING:
                met_reqs = all(s in self.candidate.skills for s in self.job.required_skills) and \
                           self.candidate.experience_years >= self.job.min_experience
                if not met_reqs:
                    reward_val = 0.2
                    reason = "Correctly rejected unqualified candidate"
                    self.done = True
                else:
                    reward_val = -0.2
                    reason = "Rejected qualified candidate"
                    self.done = True
            else:
                reward_val = -0.1
                reason = "Invalid action for screening stage"

        elif self.current_stage == Stage.INTERVIEW:
            if action.action_type == ActionType.ASK_QUESTION:
                self.questions_asked += 1
                is_relevant = any(skill.lower() in action.question.lower() for skill in self.job.required_skills)
                if is_relevant:
                    reward_val = 0.1 # Partial for relevance
                    reason = "Asked relevant technical question"
                else:
                    reward_val = -0.05
                    reason = "Asked irrelevant question"
                
                answer = self._simulate_answer(action.question)
                self.history.append({"role": "interviewer", "content": action.question})
                self.history.append({"role": "candidate", "content": answer})
                
                if self.questions_asked >= self.target_questions:
                    self.current_stage = Stage.EVALUATION
            else:
                reward_val = -0.1
                reason = "Must ask questions during interview"

        elif self.current_stage == Stage.EVALUATION:
            if action.action_type == ActionType.EVALUATE_ANSWER:
                # Deterministic check: did the candidate actually have the skills asked about?
                # For simplicity, we just look at the last question/answer
                last_q = self.history[-2]["content"]
                has_skill = any(s.lower() in last_q.lower() for s in self.candidate.skills)
                expected_score = 1.0 if has_skill else 0.4
                
                if abs(action.score - expected_score) < 0.2:
                    reward_val = 0.3
                    reason = "Accurate evaluation of candidate response"
                else:
                    reward_val = -0.1
                    reason = "Inaccurate evaluation"
                
                self.evaluations.append(action.score)
                if len(self.evaluations) >= self.target_questions:
                    self.current_stage = Stage.DECISION
            else:
                reward_val = -0.1
                reason = "Must evaluate answers"

        elif self.current_stage == Stage.DECISION:
            avg_eval = sum(self.evaluations) / len(self.evaluations) if self.evaluations else 0
            should_hire = avg_eval > 0.7
            
            if action.action_type == ActionType.HIRE:
                if should_hire:
                    reward_val = 1.0
                    reason = "Correct hiring decision"
                else:
                    reward_val = -0.5
                    reason = "Hired weak candidate"
                self.done = True
            elif action.action_type == ActionType.REJECT_INTERVIEW:
                if not should_hire:
                    reward_val = 1.0
                    reason = "Correct rejection after interview"
                else:
                    reward_val = -0.5
                    reason = "Rejected strong candidate"
                self.done = True

        # Clamp reward
        reward_val = max(0.0, min(1.0, reward_val))
        return self.state(), Reward(value=reward_val, reason=reason), self.done, {}
