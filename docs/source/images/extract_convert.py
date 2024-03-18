import re

def style_to_object(style_str):
    """Converts CSS style string to a JavaScript object (dictionary in Python terms)."""
    style_tuples = [tuple(s.split(':')) for s in style_str.split(';') if s]
    style_dict = {re.sub(r'([a-z])-([a-z])', lambda x: x.group(1) + x.group(2).upper(), k.strip()): v.strip() for k, v in style_tuples}
    return style_dict

def svg_to_jsx(svg_filepath, jsx_filepath):
    with open(svg_filepath, 'r') as svg_file:
        svg_content = svg_file.read()

    # Remove XML declaration if present
    svg_content = re.sub(r'<\?xml.*\?>', '', svg_content)

    # Remove attributes that start with 'inkscape:' or 'sodipodi:'
    svg_content = re.sub(r'\s(inkscape|sodipodi|xmlns):[^=]*="[^"]*"', '', svg_content)
    svg_content = re.sub(r'\s(ref)="[^"]*"', '', svg_content)
    svg_content = re.sub(r'\s(xml:space)="[^"]*"', '', svg_content)

    # Remove the 'sodipodi:namedview' tag entirely, including its inner content if present
    # This regex accounts for multiline tags using the re.DOTALL flag
    svg_content = re.sub(r'<sodipodi:namedview[^>]*?>.*?</sodipodi:namedview>', '', svg_content, flags=re.DOTALL)
    svg_content = re.sub(r'<metadata[^>]*?>.*?</metadata>', '', svg_content, flags=re.DOTALL)

    jsx_content = svg_content.replace('class=', 'className=')
    jsx_content = re.sub(r'([a-z])-([a-z])', lambda x: x.group(1) + x.group(2).upper(), jsx_content)
    jsx_content = re.sub(r'style="([^"]*)"', lambda match: f'style={{ {style_to_object(match.group(1))} }}', jsx_content)
    jsx_content = re.sub(r'\n\s*\n', "", jsx_content)
    jsx_content = re.sub(r'<sodipodi:namedview.*</sodipodi:namedview>', "", jsx_content)

    jsx_content = re.sub(r'id=', "className=", jsx_content)

    with open(jsx_filepath, 'w') as jsx_file:
        jsx_file.write("import {styled} from \"@linaria/react\";\n\n")
        jsx_file.write("function SvgComponent() {\n    return (")
        jsx_file.write(jsx_content.replace("svg", "Svg"))
        jsx_file.write("\n)}\n\n")
        jsx_file.write("const Svg = styled.svg`\n")
        from extract_anim import get_anim
        jsx_file.write(get_anim())
        jsx_file.write("`;\n")
        jsx_file.write("export default SvgComponent;")


# Example usage
svg_file_path = 'fiber.svg'
jsx_file_path = 'fiber.jsx'
svg_to_jsx(svg_file_path, jsx_file_path)