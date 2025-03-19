def create_mapping_file(action_list, output_file='mapping.txt'):
    """
    Create a mapping.txt file from a list of action names.

    :param action_list: List of action names (e.g., ['background', 'take_cup', 'pour_milk'])
    :param output_file: The output file path for the mapping.txt (default: 'mapping.txt')
    """
    with open(output_file, 'w') as f:
        for idx, action in enumerate(action_list):
            f.write(f"{idx} {action}\n")
    print(f"Mapping file saved to: {output_file}")

# Example usage
action_list = ['12v', '12v+', '13v', '13v+', '14v', '14v+', '21v', '21v+', '23v', '23v+', '24v', '24v+', 
               '31v', '31v+', '32v', '32v+', '34v', '34v+', '41v', '41v+', '42v', '42v+', '43v', '43v+',
               '12p', '12p+', '14p', '14p+', '21p', '21p+', '23p', '23p+', '32p', '32p+', '34p', '34p+', 
               '41p', '41p+', '43p', '43p+']
create_mapping_file(action_list, output_file='mapping.txt')
