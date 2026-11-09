import pprint
from dataclasses import asdict


def flatten_list(current_list):
    new_list = []
    for item in current_list:
        if isinstance(item, list):
            new_list.extend(flatten_list(item))
        elif isinstance(item, map):
            new_list.extend(flatten_list(list(item)))
        else:
            new_list.append(item)
    return new_list

def createElement(name, props, *children):
    if len(children) == 1 and isinstance(children[0], str):
        if props is None:
            props = {}
        props["text"] = children[0]
        children = []
    return {
        "type": name,
        "props": props,
        "children": [x for x in flatten_list(children) if x != None]
    }


def compose_props(props: dict | None, **kwargs):
    if props is None:
        props = {}
    return { **props, **kwargs }

def ComponentTreeToClassInstances(component_structure, old_instances):
    if len(old_instances) and old_instances[0].type == component_structure["type"]:
        element = old_instances[0]
        element.props = component_structure["props"]
        old_children = old_instances[1]
    else:
        element = Element(**component_structure)
        old_children = []
    children_instances = [ComponentTreeToClassInstances(x, old_children[i] if i < len(old_children) else []) for i, x in enumerate(component_structure["children"])]
    rendered = element.__render__(children_instances[0] if len(children_instances) else [])
    return [element, children_instances, rendered]

def print_instance(element, children, indent=""):
    print(indent, element)#f"{' ' * indent}{element.type}({element.props})")
    if not isinstance(element, str):
        if len(element.children_instances):
            print_instance(element.children_instances[0], element.children_instances[1], indent + " : ")
    for child in children:
        print_instance(child[0], child[1], indent + "  ")


def print_instance2(element, children, indent=""):
    if not isinstance(element, str):
        if len(element.children_instances):
            return print_instance2(element.children_instances[0], element.children_instances[1], indent + " : ")
        else:
            return {"type": element.type, "props": element.props, "children": [print_instance2(child[0], child[1], indent + "  ") for child in children]}
        #print("element", element, children)
        #return {"type": element.type, "props": element.props, "children": [print_instance2(c, element.children_instances) for c in children]}

    #print(indent, element)#f"{' ' * indent}{element.type}({element.props})")
    #if not isinstance(element, str):
    #    if len(element.children_instances):
    #        print_instance(element.children_instances[0], element.children_instances[1], indent + " : ")

    return [print_instance2(child[0], child[1], indent + "  ") for child in children]


useState = lambda x: [x, lambda x: None]

class Element:
    current_call_index = -1
    state = []

    def __init__(self, type, props, children):
        self.type = type
        self.props = props
        self.state = []
        self.children_instances = []

    def useState(self, initial_state):
        self.current_call_index += 1
        if len(self.state) <= self.current_call_index:
            self.state.append(initial_state)
        return [self.state[self.current_call_index], lambda new_value, index=self.current_call_index: self.setState(index, new_value)]

    def setState(self, index, new_state):
        self.state[index] = new_state
        rerender()

    def __repr__(self):
        return f"<{self.type if isinstance(self.type, str) else self.type.__name__} props={self.props} state={self.state} children_instances={self.children_instances}>"

    def __render__(self, children):
        global useState
        if isinstance(self.type, str):
            return {"type": self.type, "props": self.props, "children": flatten_list(children)}
        if isinstance(self.type, str):
            response = self.type
            return {"type": self.type, "props": self.props,
                    "children": [render(child) for child in response["children"]]}
        else:
            self.current_call_index = -1
            useState = self.useState
            response = self.type(compose_props(self.props, children=children))
            print("response", self.type, self.props, children)
            print("response", response)
            self.children_instances = ComponentTreeToClassInstances(response, self.children_instances)
            #print(response)
            return self.children_instances[2]
            #return render(response)

from qt_from_struct import update_ui

root_element = None
def rerender():
    global root_element, old_structure, main_layout
    element = root_element
    res = element.__render__([])
    #print_instance(element.children_instances[0], element.children_instances[1], indent=" ")
    #pprint.pprint("------")
    structure = print_instance2(element.children_instances[0], element.children_instances[1])
    #pprint.pprint(structure)
    #pprint.pprint("-.-----")
    if old_structure is not None:
        update_ui(old_structure, structure, main_layout)
        old_structure = structure
    return structure

main_layout = None
old_structure = None
def render(root, key="root"):
    global root_element, main_layout, old_structure
    element = Element(**root)
    root_element = element
    structure = rerender()

    from qt_from_struct import createQtApp
    app, main_window, main_layout = createQtApp(structure)
    old_structure = structure

    # Show the main window
    main_window.setWindowTitle("Qt Interface from Structure")
    main_window.resize(400, 300)
    main_window.show()
    app.exec_()

    return element

def render_dom(root, key="root"):
    if isinstance(root, str):
        return root
    if isinstance(root["type"], str):
        response = root
        return {"type": root["type"], "props": root["props"], "children": [render(child) for child in response["children"]]}
    else:
        response = root["type"](compose_props(root["props"], children=root["children"]))
        return render(response)


