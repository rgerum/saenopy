import react

def TodoItem(props):
    value1, setValue1 = react.useState(1)
    value2, setValue2 = react.useState(props["text"])

    return (
        create_element("QVBoxLayout", { className='my-class'}, 
            <QLabel>-></QLabel>
            <QPushButton onClick=lambda x: setValue1(value1 + 1)>!</QPushButton>
            <QLabel className='my-class'>{props["text"]}{value1}value2{value2}</QLabel>
            <QTextEdit className='my-class' onTextChanged=lambda x: setValue2(x) and print("onTextChanged", x)>
                {value2}
            </QTextEdit>
        )
    )

def TodoList(props):
    return (
        create_element("QHBoxLayout", { className='my-class'}, 
            {list(map(lambda x: create_element("TodoItem", { "text": {x}})), props["entries"])}
        )
    )

def MyComponent(props):
    value1, setValue1 = react.useState(1)
    value2, setValue2 = react.useState(0)

    return (
        create_element("QVBoxLayout", { className='my-class'}, 
            <QLabel>Hello, World!</QLabel>
            <QPushButton onClick=lambda x: setValue2(1 - value2)>Click Me</QPushButton>
            create_element("TodoList", { "entries": {["foo", "bar", "baz"] if value2 == 1 else ["foo", "va"]}})
            <QLabel>{value2}</QLabel>
            {value2 == 1 and (
                <QHBoxLayout className='my-class'>
                    <QLabel>x</QLabel>
                    <QLabel>x</QLabel>
                </QHBoxLayout>
            )}
        )
    )

x = react.render(create_element("MyComponent", {}))
