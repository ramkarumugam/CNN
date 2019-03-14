classdef trainer < handle
    
    properties
        
        net
        alpha
        desError
        J
        testJ
        trainX
        trainy
        testX
        testy
    end
    
    methods
       function obj= trainer(net,trainingParams)
            obj.net=net;
            obj.alpha=trainingParams.lr;
            obj.desError=trainingParams.desiredError;
        end      
        
        function train(obj, trainX, trainy, testX, testy)
            obj.trainX=trainX;
            obj.trainy=trainy;
            obj.testX=testX;
            obj.testy=testy;
            
            obj.J=[];
            obj.testJ=[];
            
            weights = obj.net.getParams();
            
            cost=1;
            counter=1;
            
            while cost>obj.desError 
                
                obj.net.setParams(weights);
                cost = obj.net.cost(obj.trainX, obj.trainy);
                grad = obj.net.computeGradients(obj.trainX, obj.trainy);
                obj.J = [obj.J cost];
                costEval = obj.net.cost(obj.testX, obj.testy);
                obj.testJ = [obj.testJ costEval];
                counter = counter+1;
                
                fprintf('cost: %f iteration %f \n\r',cost,counter)
                
                weights=weights-grad*(obj.alpha);
            end
        end
            
    end
end
