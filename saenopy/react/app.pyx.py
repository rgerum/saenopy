import react

def TodoItem(props):
    value1, setValue1 = react.useState(1)
    value2, setValue2 = react.useState(props["text"])

    return (
        <QVBoxLayout className='my-class'>
            <QLabel>-></QLabel>
            <QPushButton onClick=lambda: setValue1(value1 + 1)>!</QPushButton>
            <QLabel className='my-class'>{props["text"]}{value1}value2{value2}</QLabel>
            <QTextEdit className='my-class' onTextChanged=lambda x: setValue2(x) and print("onTextChanged", x)>
                {value2}
            </QTextEdit>
        </QVBoxLayout>
    )

def TodoList(props):
    return (
        <QHBoxLayout className='my-class'>
            {map(lambda x: <TodoItem text={x} />, props["entries"])}
        </QHBoxLayout>
    )

def MyComponent(props):
    value1, setValue1 = react.useState(1)
    value2, setValue2 = react.useState(0)

    return (
        <QVBoxLayout className='my-class'>
            <QLabel>Hello, World!</QLabel>
            <QPushButton onClick=lambda: setValue2(1 - value2)>Click Me</QPushButton>
            <TodoList entries={["foo", "bar", "baz"] if value2 == 1 else ["foo", "va"]} />
            <QLabel>{value2}</QLabel>
            {value2 == 1 and (
                <QHBoxLayout className='my-class'>
                    <QLabel>x</QLabel>
                    <QLabel>x</QLabel>
                </QHBoxLayout>
            )}
        </QVBoxLayout>
    )

x = react.render(<MyComponent />)
