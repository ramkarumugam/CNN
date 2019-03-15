net=imgprocess(imageArraytemp);
redimg=net.reduce(net.input);

trainx=net.reshape(redimg);

trainx=net.normalise(trainx);
I=ones(length(n),1);
train=n'+I;

trainX=[trainx(1:400,:);trainx(550:900,:)];
trainy=[train(1:400,:);train(550:900,:)];
testX=[trainx(401:549,:);trainx(901:1050,:)];
testy=[train(401:549,:);train(901:1050,:)];

%%Neural Network Parameters
%hLs=hidden Layer size, iLs = input layer size, oLs =output layer size
%h1Ls= hidden Layer 1 size, and numbered accordingly for neural networks with more than one hidden layer

y=trainy;
params.iLs=250;
params.hLs=150;
%params.h1Ls=100;
%params.h2Ls=100;
%params.h3Ls=100;
params.oLs=1;

%Feedforward
%replace 'NNwithBias' with the neural network you want to use.

net= NNwithBias(params,trainX);
yHat = forward(net,trainX);

%%Training Parameters

trainingParams.desiredError = 0.05;
trainingParams.lr =0.01;

%%Training

t = trainer(net,trainingParams);
t.train(trainX, trainy, testX, testy);
plot(t.trainJ);
plot(t.testJ)
