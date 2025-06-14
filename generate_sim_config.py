import os
import json
import openai
from utils import save_config_to_file, parse_llm_yaml, generate_random_array
import argparse

# Set your OpenAI API key (ensure you have it in your environment)
openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_config_from_llm(
    task_type,
    task_info_json="task_info.json",
    prompt_template=(
        "Respond with a filled yaml file with the following fields only:\n"
        "{task_info}\n"
        "If needed use random numbers from here: {array_0_1} for 0-1 ranges and {array_0_255} for 0-255 ranges.\n"
        "Only output the YAML, nothing else."
    ),
    model="gpt-4o"
):
    """
    task_type: string, e.g. "pick_and_place"
    task_info_json: path to JSON file with task templates
    Returns the LLM's response as a YAML config dict.
    """
    # Load task_info.json and extract the entry for the given task_type
    with open(task_info_json, "r") as f:
        task_info_data = json.load(f)
    # Assume the JSON is a dict with a "tasks" key
    task_info_str = task_info_data["tasks"].get(task_type, None)
    # print(task_info_str)
    if not task_info_str:
        raise ValueError(f"Task type '{task_type}' not found in {task_info_json}")
    
    # Generate random arrays for the task
    array_0_1, array_0_255 = generate_random_array(size=10, dtype='float', min_value=0, max_value=1), generate_random_array(size=10, dtype='int', min_value=0, max_value=256)

    # Format the prompt with the task_info string
    prompt = prompt_template.format(task_info=task_info_str, array_0_1=array_0_1, array_0_255=array_0_255)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Your task is to generate a random config file for a robot environment."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.8,
    )
    reply = response.choices[0].message.content.strip()
    # print("LLM response:\n", reply)
    # Parse YAML to dict for further use
    return parse_llm_yaml(reply)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate robot environment configs using LLM.")
    parser.add_argument("--task_type", type=str, default="push_cube_to_cube", help="Task type (e.g. sitz, move_to_cube, push_cube_to_cube)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of configurations to generate")
    parser.add_argument("--task_info_json", type=str, default="task_info.json", help="Path to task templates")

    args = parser.parse_args()

    for _ in range(args.num_episodes):
        config = get_config_from_llm(
            args.task_type,
            task_info_json=args.task_info_json,
        )
        save_config_to_file(config, args.task_type)