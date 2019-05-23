import dis
import inspect
import numpy as np
import tensorflow as tf
import time
import tqdm

import types

def rename_code_object(func, new_name):
    code = func.__code__
    return types.FunctionType(
        types.CodeType(
            code.co_argcount, code.co_nlocals,
            code.co_stacksize, code.co_flags,
            code.co_code, code.co_consts,
            code.co_names, code.co_varnames,
            code.co_filename, new_name,
            code.co_firstlineno, code.co_lnotab,
            code.co_freevars, code.co_cellvars),
        func.__globals__, new_name, func.__defaults__, func.__closure__)

class TensorFlowTheanoFunction(object):
    def __init__(self, inputs, outputs, name=None):
        self._inputs = inputs
        self._outputs = outputs

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        return tf.get_default_session().run(self._outputs, feeds)

class ProperlyWrapFunc(TensorFlowTheanoFunction):

    def __init__(self, inputs, outputs, name):
        super(ProperlyWrapFunc, self).__init__(inputs, outputs)
        renamed = rename_code_object(super(ProperlyWrapFunc, self).__call__, name)
        self.__class__ = type(name, (ProperlyWrapFunc,), {'__call__': renamed})

sess = tf.InteractiveSession()

def x(a, b):
    for i in range(10):
        c = b+10
    return np.dot(a+5, c)

#for ins in dis.get_instructions(x.__code__):
#    print(ins)
#exit()
def translateFunction(func, tf_args, np_args, tf_kwargs):
    target_func = None
    if func == int:
        target_func = int
    if func == np.einsum:
        target_func = tf.einsum
    if func == np.eye:
        target_func = tf.eye
        tf_kwargs["dtype"] = tf.float64
    if func == np.floor:
        target_func = tf.floor
    if getattr(func, "tf_jit", False) is True:
        #if getattr(func, "tensor_run", None) is not None:
        return getattr(func, "build_graph")(tf_args, np_args)
    if target_func is None:
        return func(*tf_args, **tf_kwargs)
        raise ValueError("Cannot compile func", func)
    return target_func(*tf_args, **tf_kwargs)

def print_me(target):
    try:
        return "("+str(target.shape)+")"
    except:
        return str(target)

def tfjit():
    def processFunc(__func_to_compile__):
        signature = inspect.signature(__func_to_compile__)
        parameter_names = [str(param) for param in signature.parameters]
        self = []
        def compile_func(*args, **kwargs):
            if compile_func.tensor_run is not None:
                print("*args", len(args), [print_me(a) for a in args])
                try:
                    return compile_func.tensor_run(*args, **kwargs)
                except ValueError:
                    pass
            tensor_inputs = []
            numpy_inputs = []
            for name, value in zip(parameter_names, args):
                numpy_inputs.append(value)
                if isinstance(value, np.ndarray):
                    value = tf.placeholder(dtype=tf.float64, shape=value.shape, name=name)
                    tensor_inputs.append(value)
                locals()[name] = value

            def build_graph(tensor_inputs, numpy_inputs):
                tensor_locals = {name: value for name, value in zip(parameter_names, tensor_inputs)}
                numpy_locals = {name: value for name, value in zip(parameter_names, numpy_inputs)}
                stack_tf = []
                stack_np = []

                print("Compile", compile_func.numpy_run.__name__)
                for ins in tqdm.tqdm(dis.get_instructions(compile_func.numpy_run.__code__)):
                    if len(stack_tf):
                        print(stack_tf[-1])
                    print(ins)
                    if ins.opname == "LOAD_FAST":
                        tensor_var = tensor_locals[ins.argval]
                        if getattr(tensor_var, "__replace_with__", None) is not None:
                            tensor_var = getattr(tensor_var, "__replace_with__")
                        stack_tf.append(tensor_var)
                        stack_np.append(numpy_locals[ins.argval])
                        continue
                    if ins.opname == "STORE_FAST":
                        tensor_locals[ins.argval] = tf.identity(stack_tf[-1], name=ins.argval)
                        stack_tf[-1] = tensor_locals[ins.argval]
                        stack_np[-1] = numpy_locals[ins.argval] = stack_np[-1]
                        continue
                    if ins.opname == "CALL_FUNCTION" or ins.opname == "CALL_METHOD":# or ins.opname == "CALL_FUNCTION_EX":
                        tf_args = []
                        #if ins.opname == "CALL_FUNCTION_EX":
                        #    for el in stack_tf.pop():
                        #        tf_args.append(el)
                        #    tf_args = tf_args[::-1]
                        for i in range(ins.arg):
                            tf_args.append(stack_tf.pop())
                        tf_f = stack_tf.pop()
                        if isinstance(tf_f, tuple):
                            tf_args.extend(tf_f[1:][::-1])
                            tf_f = tf_f[0]

                        np_args = []
                        #if ins.opname == "CALL_FUNCTION_EX":
                        #    for el in np_args.pop():
                        #        np_args.append(el)
                        #    np_args = np_args[::-1]
                        for i in range(ins.arg):
                            np_args.append(stack_np.pop())
                        np_f = stack_np.pop()

                        #print("Call function", np_f)
                        #print("Call function", [print_me(t) for t in np_args[::-1]])
                        stack_np.append(np_f(*np_args[::-1]))
                        #print("Call function", tf_args[::-1])
                        stack_tf.append(translateFunction(tf_f, tf_args[::-1], np_args[::-1], {}))
                        continue
                    for stack in [stack_tf, stack_np]:
                        if ins.opname == "LOAD_GLOBAL":
                            if ins.argval == "int":
                                stack.append(int)
                            else:
                                stack.append(globals()[ins.argval])
                            continue
                        if ins.opname == "LOAD_METHOD":
                            if isinstance(stack[-1], tf.Tensor) and ins.argval == "astype":
                                stack[-1] = (tf.cast, stack[-1])
                            elif isinstance(stack[-1], tf.Tensor) and ins.argval == "flatten":
                                stack[-1] = (tf.reshape, stack[-1], [-1])
                            elif isinstance(stack[-1], tf.Tensor) and ins.argval == "reshape":
                                stack[-1] = (tf.reshape, stack[-1])
                            else:
                                stack[-1] = getattr(stack[-1], ins.argval)
                            continue
                        if ins.opname == "LOAD_ATTR":
                            if isinstance(stack[-1], tf.Tensor) and ins.argval == "reshape":
                                stack[-1] = lambda *args: tf.reshape(stack[-1], *args)
                            else:
                                stack[-1] = getattr(stack[-1], ins.argval)
                            continue
                        if ins.opname == "LOAD_DEREF":
                            index = compile_func.numpy_run.__code__.co_freevars.index(ins.argval)
                            var = compile_func.numpy_run.__closure__[index].cell_contents
                            if stack == stack_tf and isinstance(var, np.ndarray):
                                var = tf.Variable(var, name=ins.argval, dtype=var.dtype)
                                sess.run(var.initializer)
                            stack.append(var)
                            continue

                        if ins.opname == "LOAD_CONST":
                            stack.append(ins.argval)
                            continue
                        if ins.opname == "BINARY_ADD":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            stack.append(TOS1 + TOS)
                            continue
                        if ins.opname == "BINARY_SUBTRACT":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            stack.append(TOS1 - TOS)
                            continue
                        if ins.opname == "BINARY_MULTIPLY":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            stack.append(TOS1 * TOS)
                            continue
                        if ins.opname == "BINARY_TRUE_DIVIDE":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            stack.append(TOS1 / TOS)
                            continue
                        if ins.opname == "COMPARE_OP":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            if ins.argval == ">":
                                stack.append(TOS1 > TOS)
                                continue
                        if ins.opname == "STORE_SUBSCR":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            TOS2 = stack.pop()
                            #print("STORE_SUBSCR", TOS1, TOS, TOS2)
                            if TOS.dtype == "bool" and isinstance(TOS, tf.Tensor):
                                w = tf.where(TOS, tf.ones_like(TOS1) * TOS2, TOS1)
                                TOS1.__replace_with__ = w
                                #TOS1.assign(tf.where(TOS, tf.zeros_like(a)))
                            else:
                                TOS1[TOS] = TOS2
                            #stack.append()
                            continue
                        if ins.opname == "BINARY_SUBSCR":
                            TOS = stack.pop()
                            TOS1 = stack.pop()
                            #print("BINARY_SUBSCR", TOS1, TOS)
                            if isinstance(TOS1, tf.Tensor) or isinstance(TOS, tf.Tensor):
                                #print("Gather")
                                stack.append(tf.gather(TOS1, TOS))
                            else:
                                #print("No gather", type(TOS1), isinstance(TOS1, tf.Tensor))
                                stack.append(TOS1[TOS])
                            continue
                        if ins.opname == "BUILD_TUPLE":
                            args = []
                            for i in range(ins.arg):
                                args.append(stack.pop())
                            stack.append(tuple(args[::-1]))
                            continue
                        if ins.opname == "RETURN_VALUE":
                            #print("Return value", stack)
                            return stack[-1]
                            break
                        #print(ins)
                        raise

            return_value = build_graph(tensor_inputs, numpy_inputs)

            #print("compiling with", tensor_inputs, return_value)
            __compiled_tensorflow_func__ = TensorFlowTheanoFunction(tensor_inputs, return_value, __func_to_compile__.__name__)
            compile_func.tensor_run = __compiled_tensorflow_func__
            compile_func.build_graph = build_graph
            writer = tf.summary.FileWriter(".", sess.graph)
            return __compiled_tensorflow_func__(*numpy_inputs)

        compile_func.tensor_run = None
        compile_func.numpy_run = __func_to_compile__
        compile_func.tf_jit = True
        return compile_func
    return processFunc
