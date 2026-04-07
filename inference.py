import os
import json
import sys
from openai import OpenAI
from my_env import InterviewEnv, Action, ActionType

def main():
    # Configuration (Use environment variables for secrets)
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    api_key = os.getenv("HF_TOKEN") # Credentials should be passed via env

    if not api_key:
         print("ERROR: HF_TOKEN environment variable not set. Please provide your Hugging Face token.")
         sys.exit(1)

    client = OpenAI(base_url=api_base, api_key=api_key)

    tasks_to_run = ["easy.json", "medium.json", "hard.json"]
    
    for task_file in tasks_to_run:
        with open(f"tasks/{task_file}", "r") as f:
            config = json.load(f)
        
        env = InterviewEnv(config)
        obs = env.reset()
        
        task_name = config["name"]
        print(f"[START] task={task_name} env=ai-interview-screening model={model_name}")
        
        step_count = 0
        total_rewards = []
        done = False
        
        while not done:
            step_count += 1
            
            # LLM Prompting Logic
            prompt = f"""
            You are an AI Hiring Manager.
            Current Stage: {obs.stage}
            Job Profile: {obs.job.title} - Requires: {', '.join(obs.job.required_skills)}
            Candidate Profile: {f'{obs.candidate.name}, Skills: {", ".join(obs.candidate.skills)}' if obs.candidate else 'Hidden in this stage'}
            Interview History: {obs.history}

            Available Actions: {list(ActionType)}

            Return ONLY a JSON object representing the next action:
            Example: {{"action_type": "ask_question", "question": "Explain your experience with Python"}}
            Example: {{"action_type": "evaluate_answer", "score": 0.8}}
            Example: {{"action_type": "shortlist_candidate"}}
            """

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": "You are a professional technical recruiter."}, {"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" } if "gpt" in model_name else None
                )
                
                raw_content = response.choices[0].message.content
                # Strip potential markdown fences
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
                
                action_data = json.loads(raw_content)
                
                # Normalize action_type to lowercase for Pydantic Enum compatibility
                if "action_type" in action_data:
                    action_data["action_type"] = action_data["action_type"].lower()
                
                action = Action(**action_data)
                obs, reward, done, info = env.step(action)
                
                total_rewards.append(reward.value)
                
                print(f"[STEP] step={step_count} action={action.action_type} reward={reward.value:.2f} done={str(done).lower()} error=null")
                
            except Exception as e:
                print(f"[STEP] step={step_count} action=unknown reward=0.00 done=true error={str(e)}")
                done = True

        success = sum(total_rewards) > 0.5
        rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
        print(f"[END] success={str(success).lower()} steps={step_count} score={sum(total_rewards):.2f} rewards={rewards_str}")

if __name__ == "__main__":
    main()
