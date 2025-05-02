import re
import json
from core.model import get_model_and_tokenizer
import pandas as pd
import numpy as np

def generate_qa_pairs(text_segment, temperature, top_p, max_pairs=5):
    """Generate QA pairs from text segment using LLaMA model."""
   
    qa_pairs = []
   
    # This would be the actual LLaMA implementation
    example_questions = [
        "Is access to applications, operating systems, databases, and network devices provisioned according to the principle of least privilege?",
        "Has management approved an access control policy, communicated it to constituents, appointed an owner to maintain it, and reviewed it?",
        "Are remote users prevented from copying data to remote non-corporate devices when using remote terminal services?",
        "Do contractual agreements specify whether third parties are permitted to resell, assign, or permit access to customer data, or the outsourcer's data, metadata, and systems, to other entities?",
        "Are inactive constituent user IDs disabled and deleted after defined periods of inactivity?"
    ]

    formatted_examples = '\n'.join(f"- {q}" for q in example_questions)

    prompt = f"""
    You are a compliance and security analyst with expertise in Governance, Risk, and Compliance (GRC) standards. Your task is to read a segment of a policy or regulatory document and generate up to {max_pairs} high-quality question-answer pairs based solely on the text.

    Each question should:
    - Relate to GRC controls, practices, or policies (e.g., access control, data protection, vendor management, incident response)
    - Be concise, unambiguous, and directly answerable from the provided text
    - Be suitable for use in a security audit or compliance checklist

    Each answer should:
    - Be factual and based only on the given text
    - Be clear and complete enough for an assessor or compliance tool

    Additionally, for each pair, include a field indicating which GRC domain or policy area it relates to (e.g., "Access Control", "Data Governance", "Incident Response").

    TEXT:
    {text_segment}

    EXAMPLES of good questions, Use questions similar to the following (don't copy, just mirror the intent:
    {formatted_examples}

    Format your response as a JSON list:
    [
        {{
            "question": "Your question here?",
            "answer": "Answer based only on the above text.",
            "grc_domain": "Relevant GRC domain (e.g., Access Control)"
        }},
        ...
    ]
    """
    
    # In a real implementation, this is where you would call the LLaMA model
    
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = generate_from_qwen(prompt)
    
    print(response)

    # Extract JSON from response
    json_match = re.search(r'\[\s*{\s*"question"', response)
    if json_match:
        json_str = response[json_match.start():]
        try:
            qa_pairs = json.loads(json_str)
        except:
            # Fallback to regex extraction if JSON parsing fails
            questions = re.findall(r'"question":\s*"([^"]+)"', json_str)
            answers = re.findall(r'"answer":\s*"([^"]+)"', json_str)
            confidences = re.findall(r'"confidence":\s*([\d.]+)', json_str)
            
            for i in range(min(len(questions), len(answers), len(confidences))):
                qa_pairs.append({
                    "question": questions[i],
                    "answer": answers[i],
                    "confidence": 1,
                    "flags": []
                })
    
    return qa_pairs


def generate_from_qwen(prompt):
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer, model = get_model_and_tokenizer(model_name)

    messages = [
        {"role": "user", "content": prompt}
    ]
    print("Applying chat template...")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating text...")
    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print(thinking_content)
    return content

