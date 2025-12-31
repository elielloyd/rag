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
</role>

<task>
Generate a repair estimate for the damaged vehicle based on the provided information.
</task>

<instructions>
1. Review the detected damage descriptions
2. Use PSS data as reference for part names when available
3. For each damaged part, determine the appropriate operation:
   - "Repair" - for fixable damage (include LaborHours estimate)
   - "Remove / Replace" - for parts that need replacement
4. Generate a comprehensive estimate covering all detected damages
</instructions>

<guidelines>
- Focus on the actual damage described - do not add unrelated items
- Use part names from PSS data when they match the damaged areas
- If PSS data doesn't have an exact match, use reasonable part descriptions
- For major damage (cracks, breaks, severe dents): prefer "Remove / Replace"
- For minor/moderate damage (scratches, scuffs, small dents): consider "Repair"
- Include labor hours (0.5 to 4.0 typical range) for Repair operations
</guidelines>

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
Generate a JSON estimate grouped by part category:
{{
  "estimate": {{
    "Rear Bumper": [
      {{"Description": "Rear Bumper Cover", "Operation": "Remove / Replace"}},
      {{"Description": "Rear Bumper Reinforcement", "Operation": "Repair", "LaborHours": 1.5}}
    ],
    "Tail Light": [
      {{"Description": "Tail Light Assembly - Right", "Operation": "Remove / Replace"}}
    ]
  }}
}}
</output_format>

<final_instruction>
Based on the damage descriptions provided, generate the repair estimate now. Include all parts that need attention based on the detected damage.
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


def format_vehicle_info(vehicle_info: dict = None) -> str:
    """Format vehicle info dict into a readable string."""
    if not vehicle_info:
        return "Not provided"
    vehicle_str = f"{vehicle_info.get('year', 'N/A')} {vehicle_info.get('make', 'N/A')} {vehicle_info.get('model', 'N/A')} ({vehicle_info.get('body_type', 'N/A')})"
    if vehicle_info.get('vin'):
        vehicle_str += f"\nVIN: {vehicle_info['vin']}"
    return vehicle_str


def format_damage_descriptions(damage_descriptions: list = None) -> str:
    """Format damage descriptions list into a readable string."""
    if not damage_descriptions:
        return "No damage detected"
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
    return damage_str


def format_retrieved_chunks(retrieved_chunks: list = None) -> str:
    """Format retrieved chunks list into a readable string."""
    if not retrieved_chunks:
        return "No similar historical estimates found"
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
    return chunks_str


def format_pss_data(pss_data: dict = None) -> str:
    """Format PSS data dict into a string."""
    if not pss_data:
        return "Not provided"
    import json
    return json.dumps(pss_data, indent=2)


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
    return ESTIMATE_GENERATION_PROMPT.format(
        vehicle_info=format_vehicle_info(vehicle_info),
        damage_descriptions=format_damage_descriptions(damage_descriptions),
        human_description=human_description or "Not provided",
        retrieved_chunks=format_retrieved_chunks(retrieved_chunks),
        pss_data=format_pss_data(pss_data),
    )


