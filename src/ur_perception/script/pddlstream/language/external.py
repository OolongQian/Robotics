from collections import Counter

from pddlstream.algorithms.common import compute_complexity
from pddlstream.language.constants import get_args, is_parameter
from pddlstream.language.conversion import values_from_objects, substitute_fact
from pddlstream.language.object import Object
from pddlstream.language.statistics import Performance, PerformanceInfo, DEFAULT_SEARCH_OVERHEAD
from pddlstream.utils import elapsed_time, get_mapping

DEBUG = 'debug'

class ExternalInfo(PerformanceInfo):
    def __init__(self, eager=False, p_success=None, overhead=None, effort=None):
        super(ExternalInfo, self).__init__(p_success, overhead, effort)
        # TODO: enable eager=True for inexpensive test streams by default
        # TODO: make any info just a dict
        self.eager = eager
        #self.complexity_fn = complexity_fn

##################################################

class Result(object):
    def __init__(self, instance, opt_index, call_index, optimistic):
        self.instance = instance
        self.opt_index = opt_index
        self.call_index = call_index
        self.optimistic = optimistic

    @property
    def external(self):
        return self.instance.external

    def get_domain(self):
        return self.instance.get_domain()

    def get_certified(self):
        raise NotImplementedError()

    def get_components(self):
        return [self]

    def get_unsatisfiable(self):
        return [self.get_components()]

    def get_tuple(self):
        raise NotImplementedError()

    def remap_inputs(self, bindings):
        raise NotImplementedError()

    def is_successful(self):
        raise NotImplementedError()

    def compute_complexity(self, evaluations):
        # Should be constant
        return compute_complexity(evaluations, self.get_domain()) + \
               self.external.get_complexity(self.call_index)

    def get_effort(self, **kwargs):
        if not self.optimistic:
            return 0  # Unit efforts?
        if self.external.is_negated():
            return 0
        # TODO: this should be the min of all instances
        return self.instance.get_effort(**kwargs)

##################################################

class Instance(object):
    _Result = None
    def __init__(self, external, input_objects):
        self.external = external
        self.input_objects = tuple(input_objects)
        self.enumerated = False
        self.disabled = False # TODO: perform disabled using complexity
        self.opt_index = 0
        self.results_history = []
        self.successes = 0
        self.opt_results = []
        self._mapping = None
        self._domain = None

    @property
    def num_calls(self):
        return len(self.results_history)

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = get_mapping(self.external.inputs, self.input_objects)
            for constant in self.external.constants:
                self._mapping[constant] = Object.from_name(constant)
        return self._mapping

    def get_mapping(self):
        return self.mapping

    @property
    def domain(self):
        if self._domain is None:
            #self._domain = substitute_expression(self.external.domain, self.get_mapping())
            self._domain = tuple(substitute_fact(atom, self.get_mapping())
                                 for atom in self.external.domain)
        return self._domain

    def get_domain(self):
        return self.domain

    def get_objects(self):
        return set(self.input_objects)

    def get_input_values(self):
        return values_from_objects(self.input_objects)

    #def is_first_call(self): # TODO: use in streams
    #    return self.online_calls == 0
    #
    #def has_previous_success(self):
    #    return self.online_success != 0

    def next_results(self, verbose=False):
        raise NotImplementedError()

    def get_results(self, start=0):
        results = []
        for index in range(start, self.num_calls):
            results.extend(self.results_history[index])
        return results

    def compute_complexity(self, evaluations):
        # Will change as self.num_calls increases
        return compute_complexity(evaluations, self.get_domain()) + \
               self.external.get_complexity(self.num_calls)

    def get_effort(self, search_overhead=DEFAULT_SEARCH_OVERHEAD):
        # TODO: handle case where resampled several times before the next search (search every ith time)
        replan_effort = self.opt_index * search_overhead  # By linearity of expectation
        effort_fn = self.external.info.effort
        if callable(effort_fn):
            return replan_effort + effort_fn(*self.get_input_values())
        return replan_effort + self.external.get_effort(search_overhead=search_overhead)

    def update_statistics(self, start_time, results):
        overhead = elapsed_time(start_time)
        successes = len([r.is_successful() for r in results])
        self.external.update_statistics(overhead, bool(successes))
        self.results_history.append(results)
        self.successes += successes

    def disable(self, evaluations, domain):
        self.disabled = True

    def enable(self, evaluations):
        self.disabled = False

##################################################

class External(Performance):
    _Instance = None
    def __init__(self, name, info, inputs, domain):
        super(External, self).__init__(name, info)
        self.inputs = tuple(inputs)
        self.domain = tuple(domain)
        for p, c in Counter(self.inputs).items():
            if not is_parameter(p):
                # AssertionError: Expected item to be a variable: q2 in (?q1 q2)
                raise ValueError('Input [{}] for stream [{}] is not a parameter'.format(p, name))
            if c != 1:
                raise ValueError('Input [{}] for stream [{}] is not unique'.format(p, name))
        parameters = {a for i in self.domain for a in get_args(i) if is_parameter(a)}
        for p in (parameters - set(self.inputs)):
            raise ValueError('Parameter [{}] for stream [{}] is not included within inputs'.format(p, name))
        for p in (set(self.inputs) - parameters):
            print('Warning! Input [{}] for stream [{}] is not covered by a domain condition'.format(p, name))
        self.constants = {a for i in self.domain for a in get_args(i) if not is_parameter(a)}
        self.instances = {}
    def is_fluent(self):
        raise NotImplementedError()
    def is_negated(self):
        raise NotImplementedError()
    def is_special(self):
        return False
    def get_complexity(self, num_calls):
        raise NotImplementedError()
    def get_instance(self, input_objects):
        input_objects = tuple(input_objects)
        assert len(input_objects) == len(self.inputs)
        if input_objects not in self.instances:
            self.instances[input_objects] = self._Instance(self, input_objects)
        return self.instances[input_objects]

##################################################

def get_procedure_fn(stream_map, name):
    if stream_map == DEBUG:
        return DEBUG
    if name not in stream_map:
        raise ValueError('Undefined external procedure: {}'.format(name))
    return stream_map[name]

def is_attribute(attribute):
    return isinstance(attribute, str) and attribute.startswith(':')

def parse_lisp_list(lisp_list):
    attributes = [lisp_list[i] for i in range(0, len(lisp_list), 2)]
    for attribute in attributes:
        if not is_attribute(attribute):
            raise ValueError('Expected an attribute but got: {}'.format(attribute))
    values = [lisp_list[i] for i in range(1, len(lisp_list), 2)]
    if len(lisp_list) % 2 != 0:
        raise ValueError('No value specified for attribute [{}]'.format(lisp_list[-1]))
    return get_mapping(attributes, values)
