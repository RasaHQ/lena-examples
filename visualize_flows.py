import logging
import os
import sys
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def transform_yaml_to_mermaid(yml_file_path):
    # Read the YAML file
    with open(yml_file_path, 'r') as file:
        yml_content = file.read()

    # Parse the YAML content
    yml_data = yaml.safe_load(yml_content)

    if not yml_data or 'flows' not in yml_data:
        log.warning("%s has no top-level 'flows' key. Skipping.", yml_file_path)
        return ""

    # Extracting the first flow
    flow_name = list(yml_data['flows'].keys())[0]
    flow = yml_data['flows'][flow_name]

    # Helper function to get step ID from a step dict
    def get_step_id(step_dict):
        return (step_dict.get('id') or step_dict.get('collect') or 
                step_dict.get('action') or step_dict.get('link') or step_dict.get('call'))
    
    # Helper function to extract NLU trigger information
    def extract_nlu_triggers(flow):
        triggers = []
        if 'nlu_trigger' not in flow:
            return triggers
        
        for trigger in flow['nlu_trigger']:
            if isinstance(trigger, dict) and 'intent' in trigger:
                intent_info = trigger['intent']
                # Handle both string and dict formats
                if isinstance(intent_info, str):
                    triggers.append(intent_info)
                elif isinstance(intent_info, dict):
                    intent_name = intent_info.get('name', 'unknown')
                    trigger_str = intent_name
                    if 'confidence_threshold' in intent_info:
                        confidence = intent_info['confidence_threshold']
                        trigger_str += f" (confidence_threshold: {confidence})"
                    triggers.append(trigger_str)
        return triggers
    
    # Helper function to find target ID in steps_list for a nested step
    def find_target_in_steps(first_step, steps_list):
        target_id = get_step_id(first_step)
        if 'set_slots' in first_step:
            slot_info = first_step['set_slots'][0] if isinstance(first_step['set_slots'], list) else first_step['set_slots']
            slot_name = list(slot_info.keys())[0] if isinstance(slot_info, dict) else 'slots'
            for sid, _, _ in steps_list:
                if sid.startswith(f"set_{slot_name}"):
                    return sid
        return target_id

    # Helper function to recursively extract steps from nested structures
    def extract_steps(steps, steps_list, parent_id=None):
        for step in steps:
            # Handle set_slots as a special case
            if 'set_slots' in step:
                # Create a unique ID for set_slots step
                slot_info = step['set_slots'][0] if isinstance(step['set_slots'], list) else step['set_slots']
                slot_name = list(slot_info.keys())[0] if isinstance(slot_info, dict) else 'slots'
                step_id = f"set_{slot_name}_{len(steps_list)}"
                # Format: "set slot_name: value"
                if isinstance(slot_info, dict):
                    slot_items = [f"{k}: {v}" for k, v in slot_info.items()]
                    label = f"set_slots<br/>{', '.join(slot_items)}"
                else:
                    label = "set_slots"
                steps_list.append((step_id, label, step))
                continue
            
            step_id = get_step_id(step)
            if not step_id:
                continue
            
            # Collect steps: always "collect: <slot_name>". Do not use description.
            # Link/call steps: show which flow they're jumping to
            # Other steps: action, question, or step_id
            if 'collect' in step:
                label = f"collect: {step_id}"
            elif 'link' in step:
                label = f"link: {step_id}"
            elif 'call' in step:
                label = f"call: {step_id}"
            else:
                label = step.get('action') or step.get('question') or step_id
            steps_list.append((step_id, label, step))
            
            # Recursively process nested steps in 'next' if they contain step objects
            if 'next' in step and isinstance(step['next'], list):
                for next_item in step['next']:
                    if isinstance(next_item, dict):
                        # Handle 'then' branch
                        if 'then' in next_item and isinstance(next_item['then'], list):
                            extract_steps(next_item['then'], steps_list, step_id)
                        # Handle 'else' branch
                        if 'else' in next_item and isinstance(next_item['else'], list):
                            extract_steps(next_item['else'], steps_list, step_id)
    
    # Build list of steps (each is one node; no subgraphs). Steps are sequential.
    steps_list = []
    extract_steps(flow['steps'], steps_list)
    
    # Begin the transformation to Mermaid
    mermaid_content = ["```mermaid", "graph TD"]

    # Add the flow's name as a level 2 heading
    mermaid_content.insert(0, f"## {flow_name}")

    # Add the flow's description after the heading
    description = f"\n{flow['description']}\n"

    # Build flow-level metadata
    metadata = []
    
    # Extract and format NLU triggers
    triggers = extract_nlu_triggers(flow)
    if triggers:
        nlu_trigger_text = "**nlu_trigger:**\n" + "\n".join(f"- {t}" for t in triggers)
        metadata.append(nlu_trigger_text)
    
    # Build flow guard metadata
    if 'if' in flow:
        if flow['if'] == False:
            metadata.append(f"**Flow Guard:** if: {flow['if']} (only triggered via call/link)")
        else:
            metadata.append(f"**Flow Guard:** if: {flow['if']}")
    
    # Build persisted slots metadata
    if 'persisted_slots' in flow:
        slots = ', '.join(flow['persisted_slots'])
        metadata.append(f"**persisted_slots:** {slots}")
    
    # Add metadata to description
    if metadata:
        for item in metadata:
            description += '\n' + item + '\n'
    
    mermaid_content.insert(1, description)

    # Check if END is referenced in the flow
    has_end = False
    for _, _, step in steps_list:
        if 'next' in step:
            if isinstance(step['next'], list):
                for next_step in step['next']:
                    if 'then' in next_step and next_step['then'] == 'END':
                        has_end = True
                        break
                    if 'else' in next_step and next_step['else'] == 'END':
                        has_end = True
                        break
            elif step['next'] == 'END':
                has_end = True
    
    # Add END node with distinctive shape only if it's referenced
    if has_end:
        mermaid_content.append("    END([END])")
    
    # One node per step (no subgraphs) — use different shapes for link/call
    for step_id, label, step in steps_list:
        if 'link' in step:
            # Hexagon shape for link (jumps without return)
            mermaid_content.append(f"    {step_id}{{{{{label}}}}}")
        elif 'call' in step:
            # Subroutine shape for call (jumps and returns)
            mermaid_content.append(f"    {step_id}[[{label}]]")
        elif 'set_slots' in step:
            # Parallelogram shape for set_slots
            mermaid_content.append(f"    {step_id}[/{label}/]")
        else:
            # Regular rectangle for other steps
            mermaid_content.append(f"    {step_id}[{label}]")

    # Sequential edges: step_i --> step_i+1 only when step has no explicit "next"
    for i in range(len(steps_list) - 1):
        curr_id, _, step = steps_list[i]
        if 'next' in step:
            continue
        next_id, _, _ = steps_list[i + 1]
        mermaid_content.append(f"    {curr_id} --> {next_id}")

    # Explicit "next" links (conditionals and branches to END or other steps)
    for step_id, _, step in steps_list:
        if 'next' not in step:
            continue
        if isinstance(step['next'], list):
            for next_step in step['next']:
                if 'if' in next_step:
                    then_val = next_step['then']
                    # Handle both string targets and list of steps
                    if isinstance(then_val, str):
                        target = then_val
                        mermaid_content.append(f"    {step_id} -->|if {next_step['if']}| {target}")
                    elif isinstance(then_val, list) and len(then_val) > 0:
                        target_id = find_target_in_steps(then_val[0], steps_list)
                        if target_id:
                            mermaid_content.append(f"    {step_id} -->|if {next_step['if']}| {target_id}")
                elif 'else' in next_step:
                    else_val = next_step['else']
                    # Handle both string targets and list of steps
                    if isinstance(else_val, str):
                        target = else_val
                        mermaid_content.append(f"    {step_id} -->|else| {target}")
                    elif isinstance(else_val, list) and len(else_val) > 0:
                        target_id = find_target_in_steps(else_val[0], steps_list)
                        if target_id:
                            mermaid_content.append(f"    {step_id} -->|else| {target_id}")
        else:
            mermaid_content.append(f"    {step_id} --> {step['next']}")

        if 'link' in step:
            mermaid_content.append(f"    {step_id} --> {step['link']}")

    mermaid_content.append("```\n\n")

    # Combine all Mermaid content
    mermaid_output = "\n".join(mermaid_content)

    return mermaid_output

def process_input(input_path, output_path=None):
    # Check if the provided path is a directory
    markdown = ""
    if os.path.isdir(input_path):
        # Walk through all subdirectories recursively
        for root, dirs, files in os.walk(input_path):
            for filename in sorted(files):
                if filename.endswith('.yml') or filename.endswith('.yaml'):
                    file_path = os.path.join(root, filename)
                    markdown += transform_yaml_to_mermaid(file_path)
        # Use provided output path or default to flows.md in input directory
        if output_path is None:
            output_path = os.path.join(input_path, "flows.md")
    # Check if the provided path is a file
    elif os.path.isfile(input_path):
        markdown = transform_yaml_to_mermaid(input_path)
        # Use provided output path or default to same directory with .md extension
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0] + ".md")
    else:
        log.error("%s is neither a valid file nor a directory.", input_path)
        return
    markdown = markdown.replace('"', '')
    if markdown.strip():
        with open(output_path, 'w') as f:
            f.write(markdown)
        log.info("Saved to %s", output_path)
    else:
        log.warning("No flow content to write.")
        return
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        log.info("Usage: python visualize_flow.py <input_path> [output_path]")
        log.info("  input_path:  Path to YAML file or directory")
        log.info("  output_path: (Optional) Path to output markdown file")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None
    process_input(input_path, output_path)