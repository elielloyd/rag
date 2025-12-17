"""Prompts for vehicle damage analysis using Gemini API."""


IMAGE_CLASSIFICATION_PROMPT = """You are an expert vehicle damage assessor. Analyze this vehicle image and classify which side/view of the vehicle it shows.

Classify the image into ONE of these categories:
- "front": Shows the front of the vehicle (headlights, front bumper, grille visible)
- "rear": Shows the rear of the vehicle (taillights, rear bumper, trunk/hatch visible)
- "left": Shows the left side of the vehicle (driver side in US/Canada)
- "right": Shows the right side of the vehicle (passenger side in US/Canada)
- "roof": Shows the roof/top of the vehicle
- "unknown": Cannot determine the vehicle side or not a vehicle image

Provide your classification with a confidence score between 0 and 1.

Respond in the exact JSON structure specified."""


DAMAGE_ANALYSIS_PROMPT = """You are an expert vehicle damage assessor analyzing images of a {year} {make} {model} ({body_type}).

The vehicle has the following approved estimate for repairs:
{approved_estimate}

Analyze the provided images showing the {side} of the vehicle and identify ALL visible damage.

For each damage found, provide:
1. **location**: Specific location on the vehicle (e.g., "Front Right Corner", "Rear Left Quarter Panel")
2. **part**: The specific part affected (e.g., "Front Bumper Cover", "Fender", "Door Panel")
3. **severity**: Rate as "Minor", "Medium", or "Major"
4. **type**: Type of damage (e.g., "Scuffing", "Scratches", "Dent", "Crack", "Broken/Shattered", "Paint Damage")
5. **start_position**: Where the damage begins (e.g., "Below headlight assembly")
6. **end_position**: Where the damage ends (e.g., "Bottom lip/valance edge")
7. **description**: Detailed description of the damage including:
   - Visual characteristics (color changes, texture, depth)
   - Extent and spread of damage
   - Impact indicators (direction, force evidence)
   - Material condition (paint layers exposed, plastic stress marks)

Be thorough and precise. Describe what you actually see in the images. Cross-reference with the approved estimate parts when applicable.

Respond in the exact JSON structure specified."""


MERGE_DAMAGE_PROMPT = """You are an expert vehicle damage assessor. Based on the following individual damage descriptions from different views of a {year} {make} {model} ({body_type}), create a comprehensive merged narrative description.

Individual damage descriptions:
{damage_descriptions}

Create a single, coherent narrative that:
1. Summarizes all damage points across the vehicle
2. Groups related damages by area
3. Describes the overall condition and damage pattern
4. Uses professional automotive terminology
5. Is suitable for an insurance claim report

The narrative should be 2-4 sentences that capture the full extent of damage.

Respond with just the merged description text, no JSON formatting."""


def get_classification_prompt() -> str:
    """Get the image classification prompt."""
    return IMAGE_CLASSIFICATION_PROMPT


def get_damage_analysis_prompt(
    year: int,
    make: str,
    model: str,
    body_type: str,
    side: str,
    approved_estimate: dict,
) -> str:
    """
    Get the damage analysis prompt with vehicle context.
    
    Args:
        year: Vehicle year
        make: Vehicle manufacturer
        model: Vehicle model
        body_type: Vehicle body type
        side: Side of vehicle being analyzed
        approved_estimate: Approved estimate operations
    
    Returns:
        The complete prompt string.
    """
    estimate_str = ""
    for part_category, operations in approved_estimate.items():
        estimate_str += f"\n{part_category}:\n"
        for op in operations:
            # Handle both dict and Pydantic model
            if hasattr(op, 'Description'):
                desc = op.Description
                operation = op.Operation
                hours = getattr(op, 'LabourHours', None)
            else:
                desc = op.get('Description', op.get('description', ''))
                operation = op.get('Operation', op.get('operation', ''))
                hours = op.get('LabourHours', op.get('LaborHours', op.get('labor_hours', '')))
            if hours:
                estimate_str += f"  - {desc}: {operation} ({hours} hours)\n"
            else:
                estimate_str += f"  - {desc}: {operation}\n"
    
    return DAMAGE_ANALYSIS_PROMPT.format(
        year=year,
        make=make,
        model=model,
        body_type=body_type,
        side=side,
        approved_estimate=estimate_str if estimate_str else "No approved estimate provided",
    )


def get_merge_damage_prompt(
    year: int,
    make: str,
    model: str,
    body_type: str,
    damage_descriptions: list[dict],
) -> str:
    """
    Get the merge damage prompt with all damage descriptions.
    
    Args:
        year: Vehicle year
        make: Vehicle manufacturer
        model: Vehicle model
        body_type: Vehicle body type
        damage_descriptions: List of damage description dictionaries
    
    Returns:
        The complete prompt string.
    """
    desc_str = ""
    for i, damage in enumerate(damage_descriptions, 1):
        desc_str += f"\n{i}. {damage.get('location', 'Unknown')} - {damage.get('part', 'Unknown')}:\n"
        desc_str += f"   Severity: {damage.get('severity', 'Unknown')}\n"
        desc_str += f"   Type: {damage.get('type', 'Unknown')}\n"
        desc_str += f"   Description: {damage.get('description', 'No description')}\n"
    
    return MERGE_DAMAGE_PROMPT.format(
        year=year,
        make=make,
        model=model,
        body_type=body_type,
        damage_descriptions=desc_str if desc_str else "No damage descriptions provided",
    )
