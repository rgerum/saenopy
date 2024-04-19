import xml.etree.ElementTree as ET

def get_anim():
    tree = ET.parse('fiber.svg')
    root = tree.getroot()

    paths = []
    objs = []
    for elem in root.findall(".//*[@id='layer2']"):
        for child in elem:
            for index, path in enumerate(child):
                if len(paths) <= index:
                    paths.append([])
                    objs.append(path)
                paths[index].append(path.attrib["d"])

    text = ""
    for j, p in enumerate(paths):
        #const {objs[j].attrib['var_name']} = styled({objs[j].attrib['var_base']})`
        anim_name = "anim_"+objs[j].attrib['id'].replace("-", "_")
        text += f"""
    & .{objs[j].attrib['id']} {{
      animation: {anim_name} 3s infinite linear;
    }}
    
    @keyframes {anim_name} {{\n"""
        for i, entry in enumerate(p):
            text += f"  {int(i*100/(len(p)-1))}% {{\n"
            text += f"    d: path(\"{entry}\");\n"
            text += "  }\n"
        text += "}\n"
    return text
