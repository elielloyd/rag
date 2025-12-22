import json

AVAILABLE_OPERATIONS = {
  1:"Remove/Replace",
  2: "Remove/Install",
  3:"Additional Labor",
  4:"Align",
  5:"Overhaul",
  6:"Refinish Only",
  7:"Access/Inspect",
  8:"Check/Adjust",
  9:"Repair",
  10:"Blend",
  16:"Paintless Repair"
}


def extract_images(images_data):
    """
    Extract only required fields from Images data
    """
    if not images_data:
        return []
    
    optimized_images = []
    
    if isinstance(images_data, list):
        for image_obj in images_data:
            optimized_image = {
                "Location": image_obj.get("Location"),
                "Callouts": []
            }
            
            callouts = image_obj.get("Callouts", [])
            for callout in callouts:
                optimized_callout = {
                    "CalloutNumber": callout.get("CalloutNumber"),
                    "PartId": callout.get("PartId")
                }
                optimized_image["Callouts"].append(optimized_callout)
            
            optimized_images.append(optimized_image)
    
    elif isinstance(images_data, dict):
        optimized_image = {
            "Location": images_data.get("Location"),
            "Callouts": []
        }
        
        callouts = images_data.get("Callouts", [])
        for callout in callouts:
            optimized_callout = {
                "CalloutNumber": callout.get("CalloutNumber"),
                "PartId": callout.get("PartId")
            }
            optimized_image["Callouts"].append(optimized_callout)
        
        optimized_images.append(optimized_image)
    
    return optimized_images 


def extract_required_pss_data(full_pss_data):
    """
    Extract only required fields from PSS data
    """
    optimized_pss = {"Categories": [] ,'SuperCategories':full_pss_data.get('SuperCategories',[])}
    categories = full_pss_data.get("Categories", [])
    for category in categories:
        optimized_category = {
            "Id": category.get("Id"),
            "Description": category.get("Description"),
            "SubCategories": []
        }
        
        subcategories = category.get("SubCategories", [])
        for subcategory in subcategories:
            optimized_subcategory = {
                "Id": subcategory.get("Id"),
                "Description": subcategory.get("Description"),
                "Parts": [],
                "Images": extract_images(subcategory.get("Images", []))
            }
            
            parts = subcategory.get("Parts", [])
            for part in parts:
                # Skip R&I and Refinish parts
                part_description = part.get("Description", "").lower()
                if "r&i" in part_description:
                    continue
                
                optimized_part = {
                    "Id": part.get("Id"),
                    "Description": part.get("Description"),
                    "PartDetails": []
                }
                
                part_details = part.get("PartDetails", [])
                for detail in part_details:
                    part_obj = detail.get("Part", {})
                    price_obj = part_obj.get("Price", {})
                    current_price = price_obj.get("CurrentPrice", 0)
                    
                    # Only include expensive parts (>$100)
                    # if current_price > 100:
                    optimized_detail = {
                        "Id": detail.get("Id"),
                        "FullDescription": detail.get("FullDescription"),
                        "Part": {
                            "Description": part_obj.get("Description"),
                            "Price": {"CurrentPrice": current_price}
                        },
                        "AvailableOperations":[]
                    }
                    for operation in detail.get("LaborOperations",[]):
                        print(operation.get("LaborOperationId",""),"LaborOperationId")
                        if AVAILABLE_OPERATIONS.get(operation.get("LaborOperationId","")):
                            optimized_detail["AvailableOperations"].append(AVAILABLE_OPERATIONS.get(operation.get("LaborOperationId")))
                    optimized_part["PartDetails"].append(optimized_detail)
                
                if optimized_part["PartDetails"]:
                    optimized_subcategory["Parts"].append(optimized_part)
            
            if optimized_subcategory["Parts"]:
                optimized_category["SubCategories"].append(optimized_subcategory)
        
        if optimized_category["SubCategories"]:
            optimized_pss["Categories"].append(optimized_category)
    
    return optimized_pss



with open("pss_subaru_copy.json", "r") as f:
    full_pss_data = json.load(f)

with open("optimized_pss.json", "w") as f:
    json.dump(extract_required_pss_data(full_pss_data), f, indent=4)
