import random, argparse, json, os
from math import log, isinf
from itertools import chain, product
from numpy import argsort, argmax, where, delete, bincount
from operator import mul
import isaac as sc
from tools import profile_execution_failure
from isaac.external.sklearn.forest import RandomForestRegressor
import optimize, tools, model
from json import encoder
import json, csv

to_catch = (sc.OperationNotSupported, sc.OclLaunchOutOfResources, sc.CudaLaunchOutOfResources, sc.MemObjectAllocationFailure, sc.InvalidWorkGroupSize, sc.OutOfHostMemory, sc.InvalidValue)

encoder.FLOAT_REPR = lambda o: format(o, '.2f')
encoder.separators = (',',':')

def unique(L):
    seen = set()
    seen_add = seen.add
    return [ x for x in L if not (x in seen or seen_add(x))]

def pow2range(a, b):
    return [2**x for x in range(a, b)]


class Tuner:

    def __init__(self, logger, device, operation, json_path):
        self.logger = logger
        self.device = device
        self.operation = operation
        self.json_path = json_path
      
    def pprint_datapoint(self, x, y):
        if self.logger:
            self.logger.info(', '.join(map(str, x)) + ': ' + str(int(max(y))) + ' ' + tools.metric_name_of(self.operation))
            
    def run(self, levels = {'BLAS1': 'intermediate', 'BLAS2':'intermediate', 'BLAS3':'intermediate'}): 
        for key, level in levels.iteritems():
            assert key in ['BLAS1', 'BLAS2', 'BLAS3']
            assert level in ['simple', 'intermediate', 'full']
        
        device = self.device
        operation = self.operation
        context = sc.driver.context(device)
        
        if self.logger:
            self.logger.info('Now tuning ' + operation.__name__.replace('_','-').upper() + '...')
         
        sizes = {}
        #BLAS1 training sizes
        if levels['BLAS1']=='simple':
            blas1_sizes = [(1e7,)]
        elif levels['BLAS1']=='intermediate':
            blas1_sizes = [(x,) for x in tools.expspace(1e3, 1e8, 10)]
        else:
            blas1_sizes = [(x,) for x in tools.expspace(1e3, 1e8, 30)] 
        sizes[sc.templates.axpy] = blas1_sizes
        sizes[sc.templates.dot] = blas1_sizes

        #BLAS2 training sizes
        if levels['BLAS2']=='simple':
            blas2_sizes = [(1536, 1536)]
        elif levels['BLAS2']=='intermediate':
            blas2_sizes =  [(1000,256), 
                            (4096,256), 
                            (256, 1000), 
                            (256, 4096),
                            (169,256),
                            (169, 384), 
                            (729,256), 
                            (3025,96)]
        else:
            blas2_sizes = product(pow2range(4,17), pow2range(4,17))
        sizes[sc.templates.ger] = blas2_sizes
        sizes[sc.templates.gemv_n] = blas2_sizes
        sizes[sc.templates.gemv_t] = blas2_sizes

        #BLAS3 training sizes
        if levels['BLAS3']=='simple':
            blas3_sizes = [(1536,1536,1536)]
        elif levels['BLAS3']=='intermediate':
            blas3_sizes = [(32, 32, 16000),
                           (3025,96,363),
                           (729,128,1200),
                           (169,384,2304),
                           (169,192,1728),
                           (169,128,1728),
                           (169,1728,128),
                           (169,1728,192),
                           (169,2304,384),
                           (729,1200,128),
                           (1728,128,169), 
                           (1728,192,169),
                           (2304,384,169),
                           (1200,128,729),
                           (363,96,3025)]
        elif levels['BLAS3']=='full':
            blas3_sizes = product(pow2range(5, 12), pow2range(5, 12), pow2range(5, 15))
        sizes[sc.templates.gemm_nn]     = blas3_sizes
        sizes[sc.templates.gemm_tn]     = blas3_sizes
        sizes[sc.templates.gemm_nt]     = blas3_sizes
        sizes[sc.templates.gemm_tt]     = blas3_sizes
            
        #Remove duplicates
        sizes = unique(list(sizes[operation]))
        sizes = [x for x in sizes if 1e-4 <= tools.memory_footprint(operation, x) <= 1e-1]
        
        #Training data
        performance = tools.metric_of(operation)
        profiles, X, Y = [], [], []
        
        #Restore previous run
        savepath = os.path.join('save', operation.__name__)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        try:
            with open(os.path.join(savepath, 'X.csv')) as f:
                X = [tuple(map(int, row)) for row in csv.reader(f, delimiter=',')]
                
            with open(os.path.join(savepath, 'Y.csv')) as f:
                Y = [map(float, row) for row in csv.reader(f, delimiter=',')]
            
            with open(os.path.join(savepath, 'profiles.csv')) as f:
                def mmap(x):
                    if x=='FETCH_FROM_LOCAL':
                        return sc.templates.fetching_policy_type.FETCH_FROM_LOCAL
                    if x=='FETCH_FROM_GLOBAL_CONTIGUOUS':
                        return sc.templates.fetching_policy_type.FETCH_FROM_GLOBAL_CONTIGUOUS
                    if x=='FETCH_FROM_GLOBAL_STRIDED':
                        return sc.templates.fetching_policy_type.FETCH_FROM_GLOBAL_STRIDED
                    return int(x)
                profiles = [map(mmap,row) for v in row for row in csv.reader(f, delimiter=',')]
        except:
            pass
        
        for idx, x in enumerate(sizes):
            if x in X:
                self.pprint_datapoint(x, Y[X.index(x)])
                continue
            idx = len(X)
            nparams = len(profiles)
            tree, operands = tools.tree_of(operation, x, context)
            #Check if the current best prediction is not a local optimum
            if idx==0:
                tune = True
                predicted = None
            else:
                if nparams==1:
                    predicted = profiles[0]
                else:
                    clf = RandomForestRegressor(min(10, idx+1), max_depth=min(10, idx+1)).fit(X, Y)
                    #clf, nrmse = model.train(X, Y, profiles)
                    predperf = clf.predict(x)[0]
                    best = (-predperf).argsort()[:5]
                    perf = []
                    for b in best:
                        try:
                            perf += [performance(x, tools.benchmark(operation, profiles[b], tree))]
                        except (sc.OperationNotSupported, sc.LaunchOutOfResources, sc.MemObjectAllocationFailure):
                            pass
                    predicted = profiles[best[argmax(perf)]]
                tune = not optimize.is_local_optimum(predicted, operation, x, context)     
                #tune = True
            #Retune if necessary
            if tune:
                #new = optimize.exhaustive(operation, x, context)
                optimizer = optimize.GeneticOptimizer(self.logger, naccept=1000, niter=1000, cxpb=.4, mutpb=.4, popsize=20)
                new = optimizer.run(operation, x, context, prior=predicted)[0]
                if new not in profiles:
                    profiles.append(new)
                    if idx > 0:
                        for xx,yy in zip(X, Y):
                            _tree, _operands = tools.tree_of(operation, xx, context)
                            try:
                                time = tools.benchmark(operation, new, _tree)
                                perf = performance(xx, time)
                            except profile_execution_failure:
                                perf = 0
                            yy.append(0 if isinf(perf) else perf)
            #Update dataset
            y = []
            fastest = max(predperf) if nparams > 1 else None
            for ip, p in enumerate(profiles):
                try:
                    perf = 0 if fastest and ip < nparams and predperf[ip]/fastest < .1 else performance(x,tools.benchmark(operation, p, tree))
                except profile_execution_failure:
                    perf = 0
                y.append(0 if isinf(perf) else perf)
            X.append(x)
            Y.append(y)
            
            for (fname, data) in zip(['X.csv', 'Y.csv', 'profiles.csv'], [X, Y, profiles]):
                with open(os.path.join(savepath, fname), 'wb') as f:
                    csv.writer(f).writerows(data)
            
            #Update logging
            self.pprint_datapoint(x, y)

        
        #Export to JSON
        json_path = tools.sanitize(device.name) + '.json' if not self.json_path else self.json_path
        if os.path.isfile(json_path):
            json_data = json.load(open(json_path, 'r'))
        else:
            json_data = {}
            json_data["version"] = "1.0"
        operation_name = operation.__name__
        if operation_name not in json_data:
            json_data[operation_name] = {}
        json_data[operation_name]['float32'] = {}
        D = json_data[operation_name]['float32']
        if len(profiles) > 1:
            clf, nrmse = model.train(X, Y, profiles)
            D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                                'children_right': e.tree_.children_right.tolist(),
                                'threshold': e.tree_.threshold.astype('float64').tolist(),
                                'feature': e.tree_.feature.astype('float64').tolist(),
                                'value': e.tree_.value[:,:,0].astype('float64').tolist()} for e in clf.estimators_]
        D['profiles'] = [map(int, x) for x in profiles]
        json.dump(json_data, open(json_path,'w'))
