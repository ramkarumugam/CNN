classdef imgprocess < handle
    properties
        input
        convimg
        redimg
        redimg1
        hor
        ver
        i
        j
        k       
    end
    methods
        function obj=imgprocess(imageArray)
            obj.input=imageArray;
            [obj.i, obj.j, obj.k]=size(obj.input);
            
            %Input your desired kernel
            obj.hor=[1 2 1; 0 0 0; -1 -2 -1];
            obj.ver=obj.hor';
            obj.convimg=zeros(obj.i,obj.j,obj.k);
            obj.redimg=[];
            
        end
        function out=convolution(obj,input)
            z=1;
            for z=1:obj.k;
                temph=conv2(obj.input(:,:,z),obj.hor,'same');
                tempv=conv2(obj.input(:,:,z),obj.ver,'same');
                temp=sqrt((temph.^2)+(tempv.^2));
                obj.convimg(:,:,z)=temp;
                imshow(temp);
            end
            out=obj.convimg;
        end
        function out=reduce(obj, input);
            obj.convimg=obj.convolution(input);
            z=1;
            
            %%denominator-should be a divisor of both pixel width and height
            %%image will get reduced by a factor of square of denominator. 
            den=4;
            skip=den-1;
            
            for z=1:obj.k;
                x=1;
                y=1;
                while x<obj.i
                    n=1;
                    y=1;
                    while y<obj.j
                        m=(x+skip)/den;
                        n=(y+skip)/den;
                        obj.redimg(m,n,z)=mean(mean(obj.convimg(x:x+skip,y:y+skip,z)));
                        y=y+den;
                        n=n+1;
                    end
                    x=x+den;
                    m=m+1;
                end
            end
            out=obj.redimg;
        end
        
        function out=reshape(obj,image);
           X=[];
           for x=1:obj.k;
               X(x,:)=reshape(image(:,:,x), 1, numel(image(:,:,x)));
           end
           out=X;
        end
        function out=normalise(obj,trainX);
           [x,y]=size(trainX);
           mu=mean(trainX);
           mi=min(trainX);
           ma=max(trainX);
           m=1;
           norm=zeros(x,y);
           for m=1:x;
               norm(m,:)=(trainX(m,:)-mu)./(ma-mi);
           end
           out=norm;
        end
       
    end
end
