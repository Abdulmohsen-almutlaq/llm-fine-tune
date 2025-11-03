def extract_fields(example: dict, instruction_key: str = None, output_key: str = None, hash_key: str = None) -> dict:
    if instruction_key is None:
        for key in ["instruction", "question", "prompt", "input", "text"]:
            if key in example:
                instruction_key = key
                break
    
    if output_key is None:
        for key in ["output", "answer", "response", "label", "completion"]:
            if key in example:
                output_key = key
                break
    
    if hash_key is None:
        for key in ["hash", "domain", "category", "type"]:
            if key in example:
                hash_key = key
                break
    
    instruction = example.get(instruction_key, "")
    output = example.get(output_key, "")
    hash_value = example.get(hash_key, "unknown")
    
    return {
        "instruction": instruction,
        "output": output,
        "hash": hash_value
    }


def format_data(examples: list, instruction_key: str = None, output_key: str = None, hash_key: str = None) -> list:
    formatted = []
    for ex in examples:
        fields = extract_fields(ex, instruction_key, output_key, hash_key)
        hash_prefix = f"[hash: {fields['hash']}]"
        text = f"{hash_prefix} User: {fields['instruction']}\nAssistant: {fields['output']}"
        formatted.append({"text": text})
    return formatted
