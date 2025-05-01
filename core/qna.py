from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import re
import json

def generate_qa_pairs(text_segment, temperature, top_p, max_pairs=5):
    """Generate QA pairs from text segment using LLaMA model."""
   
    qa_pairs = []

    example_question = [
    "Is access to applications, operating systems, databases, and network devices provisioned according to the principle of least privilege?",
    "Has management approved an access control policy, communicated it to constituents, appointed an owner to maintain it, and reviewed it?",
    "Are remote users prevented from copying data to remote non-corporate devices when using remote terminal services?",
    "Do contractual agreements specify whether third-parties are permitted to resell, assign, or permit access to customer data, or the outsourcer's data, metadata, and systems, to other entities?",
    "Are inactive constituent user IDs disabled and deleted after defined periods of inactivity?"
    ]

    formatted_examples = '\n'.join(f"- {q}" for q in example_question)
    # This would be the actual LLaMA implementation
    prompt =f"""
    You are an expert at analyzing security and compliance documents. Given the following text, generate up to {max_pairs} pairs of relevant question-answer focusing on security and compliance controls, practices, and requirements.

    TEXT:
    {text_segment}

    EXAMPLES:
    {formatted_examples}

    Format your answer like this:
    [
        {{
            "question": "Your question here?",
            "answer": "Answer from the text."
        }},
        ...
    ]
      """

    
    # In a real implementation, this is where you would call the LLaMA model
    
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = generate_from_qwen(prompt)

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
                })
    
    return qa_pairs


def generate_from_qwen(prompt):
    model_name = "Qwen/Qwen3-0.6B"

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print(f"Model {model_name} loaded successfully!")
    messages = [
        {"role": "user", "content": prompt}
    ]
    print("Applying chat template...")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("Generating text...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Output:\n", output_text)

    # Optional: extract only the assistant's answer (after user prompt)
    if "Assistant:" in output_text:
        output_text = output_text.split("Assistant:")[-1].strip()

    return output_text

