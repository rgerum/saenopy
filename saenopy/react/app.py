import react

def TodoItem(props):
    value1, setValue1 = react.useState(1)
    value2, setValue2 = react.useState(props["text"])
    return react.createElement(
        'QVBoxLayout',
        { "className": 'my-class' },
        react.createElement('QLabel', None, '->'),
        react.createElement(
            'QPushButton',
            { "onClick": lambda x: setValue1(value1 + 1) },
            '!'
        ),
        react.createElement(
        'QLabel',
        {"className": 'my-class'},
        props["text"]+f"{value1}value2{value2}",
        ),
        react.createElement(
            'QTextEdit',
            {"className": 'my-class', "onTextChanged": lambda x: setValue2(x) and print("onTextThance", x)},
            value2,
        ),
    )

def TodoList(props):

    return react.createElement(
        'QHBoxLayout',
        {"className": 'my-class'},
        map(lambda x: react.createElement(TodoItem, {"text": x}), props["entries"]),
        )

def MyComponent(props):
    value1, setValue1 = react.useState(1)
    value2, setValue2 = react.useState(0)
    return react.createElement(
        'QVBoxLayout',
        { "className": 'my-class' },
        react.createElement('QLabel', None, 'Hello, World!'),
        react.createElement(
            'QPushButton',
            { "onClick": lambda x: setValue2(1-value2) },
            'Click Me'
        ),
        react.createElement(
            TodoList,
            { "entries": ["foo", "bar", "baz"] if value2 == 1 else ["foo", "va"]},
        ),
        react.createElement("QLabel", None, f"{value2}"),
        value2 == 1 and react.createElement(
            'QHBoxLayout',
            {"className": 'my-class'},
            react.createElement("QLabel", None, "x"),
            react.createElement("QLabel", None, "x"),
        ),
    )

x = react.render(
    react.createElement(MyComponent, None),
);