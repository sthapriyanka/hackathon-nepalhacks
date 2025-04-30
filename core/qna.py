from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import re
import json

def generate_qa_pairs(text_segment, temperature, top_p, max_pairs=5):
    """Generate QA pairs from text segment using LLaMA model."""
   
    qa_pairs = []
   
    # This would be the actual LLaMA implementation
    prompt = f"""
    You are an expert at analyzing security and compliance documents and extracting question-answer pairs.
    Given the following text from a document, create {max_pairs} clear question-answer pairs related to security policies, 
    compliance requirements, and standards. Focus on extracting factual information that would be useful for security assessments.
    
    TEXT:
    {text_segment}
    
    For each pair, provide:
    1. A clear, direct question about a security policy, requirement, or practice
    2. A concise but complete answer based only on the text
    3. A confidence score between 0 and 1
    4. Any flags for ambiguity or incompleteness
    
    Format your response as JSON:
    [
        {{
            "question": "Question text here?",
            "answer": "Answer text here.",
            "confidence": 0.85,
            "flags": ["flag 1", "flag 2"]
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
                    "confidence": float(confidences[i]),
                    "flags": []
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

