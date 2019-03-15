classdef  NNwithBias < handle
    
    properties
        iLs
        hLs
        oLs
        W1
        W2
        z2
        a2
        z3
        a3
        y
        b1
        b2
        temp
        I
    end
    
    methods
        function obj= NNwithBias(params, X)
            obj.iLs=params.iLs;
            obj.hLs=params.hLs;
            obj.oLs=params.oLs;
                        
            obj.W1=rand(obj.iLs,obj.hLs);
            obj.W2=rand(obj.hLs,obj.oLs);
            obj.b1=ones(1,obj.hLs);
            obj.b2=ones(1,obj.oLs);
            
                      
        end
       
        function out=sigmoid(obj, z)
            out=1./(1+exp(-z));
        end
        function out = inputsize(obj, X)
            [numinput, n]=size(X);
            out=numinput;
        end
        
        function out = forward(obj, X)
            
            numinput=obj.inputsize(X);
            obj.I=ones(numinput, 1);
            obj.temp=X
             
            obj.z2 = X*obj.W1+obj.I*obj.b1;
            obj.a2 = obj.sigmoid(obj.z2);
            obj.z3 = obj.a2*obj.W2+obj.I*obj.b2;
            obj.a3 = obj.sigmoid(obj.z3);
            obj.y = obj.a3;
            out=obj.y;
        end
        
             %%%obj.y
        
        %backward functions
        
        function J = cost(obj, X, y)
            obj.y=obj.forward(X);
            J = 0.5*sum((y-obj.y).^2);
        end
        
        function out=sigmoidprime(obj, z)
            out=exp(-z)./((1+exp(-z).^2));
        end
        
        function[dJdW1, dJdW2, dJdb1, dJdb2] = costprime(obj, X, y)
            obj.y=obj.forward(X);
            delta3=(-(y-obj.y).*obj.sigmoidprime(obj.z3));
            dJdW2=obj.a2'*delta3;
            dJdb2=((-(y-obj.y)')*obj.sigmoidprime(obj.z3));
                       
            delta2=(delta3.*obj.W2').*obj.sigmoidprime(obj.z2);
            dJdW1=X'*delta2;
            
            dJdb1=(-(y-obj.y).*obj.sigmoidprime(obj.z3))'*obj.sigmoidprime(obj.z2);
            obj.temp=dJdb1;
        end
        function out = computeGradients (obj, X, y)
            dJdW1 = [];
            dJdW2 = [];
            dJdb1 = [];
            dJdb2 = [];
            [dJdW1, dJdW2, dJdb1, dJdb2] = obj.costprime(X,y);
            out = [reshape(dJdW1,1,numel(dJdW1)) reshape(dJdW2,1,numel(dJdW2)) reshape(dJdb1, 1, numel(dJdb1)) reshape(dJdb2, 1, numel(dJdb2))];
        
        end

        
        function out = getParams(obj)
            % get W1 and W2 rolled into vector
           w1f=reshape(obj.W1,1,numel(obj.W1));
            w2f=reshape(obj.W2,1,numel(obj.W2));
            b1f=reshape(obj.b1, 1, numel(obj.b1));
            b2f=reshape(obj.b2, 1, numel(obj.b2));
            out=[w1f w2f b1f b2f];
            
        end
        
        function obj = setParams(obj,weights)
            
            %Set W1 and W2 using single parameter vector
            W1_start = 1;
            W1_end = obj.hLs*obj.iLs;
            obj.W1 = reshape(weights(W1_start:W1_end), [obj.iLs, obj.hLs]);
            
            W2_end = W1_end + obj.hLs*obj.oLs;
            obj.W2 = reshape(weights(W1_end+1:W2_end),[obj.hLs, obj.oLs]);
            b1_end= W2_end+obj.hLs;
            obj.b1 = reshape(weights(W2_end+1:b1_end), [1, obj.hLs]);
            b2_end=b1_end+obj.oLs;
            obj.b2= reshape(weights(b1_end+1:b2_end), [1, obj.oLs]);
        end
    end
end

