"""Prompts for RAG (Retrieval Augmented Generation) pipeline."""


DAMAGE_DETECTION_PROMPT = """<role>
You are an expert vehicle damage assessor specializing in automotive collision analysis.
</role>

<task>
Analyze the provided vehicle image to detect and document any visible damage.
</task>

<instructions>
1. Determine which side/view of the vehicle the image shows (front, rear, left, right, roof, or unknown)
2. Identify if there is ANY visible damage
3. For each damage found, document with precision:
   - location: Specific location on the vehicle (e.g., "Front Right Corner", "Rear Left Quarter Panel")
   - part: The specific part affected (e.g., "Front Bumper Cover", "Fender", "Door Panel")
   - severity: Rate as "Minor", "Medium", or "Major"
   - type: Type of damage (Scuffing, Scratches, Dent, Crack, Broken/Shattered, Paint Damage)
   - start_position: Where the damage begins
   - end_position: Where the damage ends
   - description: Detailed description including visual characteristics, extent, and material condition
</instructions>

<constraints>
- Only report damage you can actually see in the image
- Be thorough and precise
- Do not assume or invent damage that is not visible
</constraints>

<output_format>
Respond in the exact JSON structure specified by the schema.
</output_format>"""


DAMAGE_DETECTION_WITH_CONTEXT_PROMPT = """<role>
You are an expert vehicle damage assessor specializing in automotive collision analysis.
</role>

<context>
Vehicle: {year} {make} {model} ({body_type})
{human_description_context}
</context>

<task>
Analyze the provided vehicle image to detect and document any visible damage.
</task>

<instructions>
1. Determine which side/view of the vehicle the image shows (front, rear, left, right, roof, or unknown)
2. Identify if there is ANY visible damage
3. For each damage found, document with precision:
   - location: Specific location on the vehicle
   - part: The specific part affected
   - severity: Rate as "Minor", "Medium", or "Major"
   - type: Type of damage (Scuffing, Scratches, Dent, Crack, Broken/Shattered, Paint Damage)
   - start_position: Where the damage begins
   - end_position: Where the damage ends
   - description: Detailed description including visual characteristics, extent, and material condition
</instructions>

<constraints>
- Only report damage you can actually see in the image
- Be thorough and precise
- Cross-reference with the human description if provided, but prioritize what you actually observe
- Do not assume or invent damage that is not visible
</constraints>

<output_format>
Respond in the exact JSON structure specified by the schema.
</output_format>"""


ESTIMATE_GENERATION_PROMPT = """<role>
You are an expert automotive estimator specializing in collision repair estimates.
You are precise, analytical, and use only verified data sources.
</role>

<instructions>
1. **Analyze**: Review the detected damage descriptions carefully
2. **Match**: Find corresponding parts in the PSS data that match the damaged areas
3. **Decide**: For each damaged part, determine if it needs "Repair" or "Remove / Replace"
4. **Output**: Generate the estimate using exact part descriptions from PSS data
</instructions>

<constraints>
- Generate estimates ONLY for damage that was actually detected - do not invent or assume additional damage
- USE PSS DATA AS THE PRIMARY SOURCE for part names, descriptions, and categories
- Operation types are limited to:
  - "Repair" - includes LaborHours field
  - "Remove / Replace" - NO LaborHours field
- LaborHours field is ONLY included when Operation is "Repair"
</constraints>

<prioritization>
1. PRIORITIZE damages with "Major" or "Medium" severity over "Minor" damages
2. Focus on structural and safety-critical parts first
3. Group related damages by part category
4. For extensive damage (cracks, breaks, shattered): prefer "Remove / Replace"
5. For surface damage only (scuffs, scratches, minor dents): consider "Repair"
</prioritization>

<context>
<vehicle_info>
{vehicle_info}
</vehicle_info>

<detected_damage>
{damage_descriptions}
</detected_damage>

<human_description>
{human_description}
</human_description>

<historical_estimates>
{retrieved_chunks}
</historical_estimates>

<pss_data>
{pss_data}
</pss_data>
</context>

<output_format>
Generate a JSON estimate grouped by part category with this structure:
{{
  "estimate": {{
    "Category Name": [
      {{"Description": "Part description from PSS", "Operation": "Repair", "LaborHours": 1.5}},
      {{"Description": "Part description from PSS", "Operation": "Remove / Replace"}}
    ]
  }}
}}
</output_format>

<final_instruction>
Based on the detected damage and PSS data above, generate the repair estimate now.
</final_instruction>"""


def get_damage_detection_prompt() -> str:
    """Get the basic damage detection prompt."""
    return DAMAGE_DETECTION_PROMPT


def get_damage_detection_with_context_prompt(
    year: int = None,
    make: str = None,
    model: str = None,
    body_type: str = None,
    human_description: str = None,
) -> str:
    """
    Get damage detection prompt with vehicle context.
    
    Args:
        year: Vehicle year
        make: Vehicle manufacturer
        model: Vehicle model
        body_type: Vehicle body type
        human_description: Human-provided damage description
    
    Returns:
        The complete prompt string.
    """
    if not all([year, make, model, body_type]):
        return DAMAGE_DETECTION_PROMPT
    
    human_context = ""
    if human_description:
        human_context = f"The owner has described the damage as: \"{human_description}\"\n\nUse this as context but verify against what you observe."
    
    return DAMAGE_DETECTION_WITH_CONTEXT_PROMPT.format(
        year=year,
        make=make,
        model=model,
        body_type=body_type,
        human_description_context=human_context,
    )


def get_estimate_generation_prompt(
    vehicle_info: dict = None,
    damage_descriptions: list = None,
    human_description: str = None,
    retrieved_chunks: list = None,
    pss_data: dict = None,
) -> str:
    """
    Get the estimate generation prompt with all context.
    
    Args:
        vehicle_info: Vehicle information dict
        damage_descriptions: List of damage descriptions from detection
        human_description: Human-provided damage description
        retrieved_chunks: Similar chunks retrieved from Qdrant
        pss_data: Parts and Service Standards data
    
    Returns:
        The complete prompt string.
    """
    # Format vehicle info
    vehicle_str = "Not provided"
    if vehicle_info:
        vehicle_str = f"{vehicle_info.get('year', 'N/A')} {vehicle_info.get('make', 'N/A')} {vehicle_info.get('model', 'N/A')} ({vehicle_info.get('body_type', 'N/A')})"
        if vehicle_info.get('vin'):
            vehicle_str += f"\nVIN: {vehicle_info['vin']}"
    
    # Format damage descriptions
    damage_str = "No damage detected"
    if damage_descriptions:
        damage_str = ""
        for i, damage in enumerate(damage_descriptions, 1):
            if isinstance(damage, dict):
                damage_str += f"\n{i}. **{damage.get('part', 'Unknown Part')}** at {damage.get('location', 'Unknown Location')}\n"
                damage_str += f"   - Severity: {damage.get('severity', 'Unknown')}\n"
                damage_str += f"   - Type: {damage.get('type', 'Unknown')}\n"
                damage_str += f"   - Position: {damage.get('start_position', 'N/A')} to {damage.get('end_position', 'N/A')}\n"
                damage_str += f"   - Description: {damage.get('description', 'N/A')}\n"
            else:
                damage_str += f"\n{i}. {damage}\n"
    
    # Format human description
    human_str = "Not provided"
    if human_description:
        human_str = human_description
    
    # Format retrieved chunks
    chunks_str = "No similar historical estimates found"
    if retrieved_chunks:
        chunks_str = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunks_str += f"\n### Historical Estimate {i} (Similarity: {chunk.get('score', 0):.2f})\n"
            chunks_str += f"**Vehicle**: {chunk.get('vehicle_info', {}).get('year', 'N/A')} {chunk.get('vehicle_info', {}).get('make', 'N/A')} {chunk.get('vehicle_info', {}).get('model', 'N/A')}\n"
            chunks_str += f"**Side**: {chunk.get('side', 'N/A')}\n"
            chunks_str += f"**Damage Description**: {chunk.get('content', 'N/A')}\n"
            
            if chunk.get('approved_estimate'):
                chunks_str += "**Approved Operations**:\n"
                for category, operations in chunk.get('approved_estimate', {}).items():
                    chunks_str += f"  - {category}:\n"
                    for op in operations:
                        desc = op.get('Description', 'N/A')
                        operation = op.get('Operation', 'N/A')
                        hours = op.get('LabourHours', '')
                        if hours:
                            chunks_str += f"    - {desc}: {operation} ({hours} hrs)\n"
                        else:
                            chunks_str += f"    - {desc}: {operation}\n"
    
    # Format PSS data
    pss_str = "Not provided"
    if pss_data:
        import json
        pss_str = json.dumps(pss_data, indent=2)
    
    return ESTIMATE_GENERATION_PROMPT.format(
        vehicle_info=vehicle_str,
        damage_descriptions=damage_str,
        human_description=human_str,
        retrieved_chunks=chunks_str,
        pss_data=pss_str,
    )


