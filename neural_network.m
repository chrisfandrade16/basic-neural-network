function y = activate(x,W,b)
    y = 1./(1+exp(-(W*x+b)));
end

function netbp_full    
    load("dataset.mat", "X", "Y");
    [numberOfRows, numberOfColumns] = size(X);
    x1 = X(:, 1);
    x2 = X(:, 2);
    y = transpose(Y);
    numberOfPoints = numberOfRows;
    numberOfInputs = 2;
    numberOfNeuronsLayer2 = 10;
    numberOfNeuronsLayer3 = 10;
    numberOfOutputs = 2;
    numberOfIterations = 0;
    maximumNumberOfIterations = 1e6;
    learningRate = 0.08;
    initialWeightAndBiasMultiplier = 5;
    allCosts = zeros(maximumNumberOfIterations,1);
    allAccuracies = zeros(maximumNumberOfIterations,1);
    firstHalfOfPointsRange = 1:(numberOfPoints/2);
    secondHalfOfPointsRange = ((numberOfPoints/2) + 1):numberOfPoints;
    
    figure(1)
    clf
    a1 = subplot(1,1,1);
    plot(x1(firstHalfOfPointsRange),x2(firstHalfOfPointsRange),'ro','MarkerSize',12,'LineWidth',4)
    hold on
    plot(x1(secondHalfOfPointsRange),x2(secondHalfOfPointsRange),'bx','MarkerSize',12,'LineWidth',4)
    a1.XTick = [0 1];
    a1.YTick = [0 1];
    a1.FontWeight = 'Bold';
    a1.FontSize = 16;
    xlim([0,1])
    ylim([0,1])
    
    rng(5000);
    W2 = initialWeightAndBiasMultiplier*randn(numberOfNeuronsLayer2,numberOfInputs);
    W3 = initialWeightAndBiasMultiplier*randn(numberOfNeuronsLayer3,numberOfNeuronsLayer2);
    W4 = initialWeightAndBiasMultiplier*randn(numberOfOutputs,numberOfNeuronsLayer3);
    b2 = initialWeightAndBiasMultiplier*randn(numberOfNeuronsLayer2,1);
    b3 = initialWeightAndBiasMultiplier*randn(numberOfNeuronsLayer3,1);
    b4 = initialWeightAndBiasMultiplier*randn(numberOfOutputs,1);
    
    for counter = 1:maximumNumberOfIterations
        k1 = randi(numberOfPoints);
        k2 = randi(numberOfPoints);
        k3 = randi(numberOfPoints);
        k4 = randi(numberOfPoints);
        x = [x1(k1) x1(k2) x1(k3) x1(k4); x2(k1) x2(k2) x2(k3) x2(k4)];
        % Forward pass
        a2 = activate(x,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        % Backward pass
        delta4 = a4.*(1-a4).*(a4-y(:,[k1 k2 k3 k4]));
        delta3 = a3.*(1-a3).*(W4'*delta4);
        delta2 = a2.*(1-a2).*(W3'*delta3);
        % Gradient step
        W2 = W2 - learningRate*delta2*x';
        W3 = W3 - learningRate*delta3*a2';
        W4 = W4 - learningRate*delta4*a3';
        b2 = b2 - learningRate*delta2;
        b3 = b3 - learningRate*delta3;
        b4 = b4 - learningRate*delta4;
        % Monitor progress
        [newcost, newaccuracy] = cost(W2,W3,W4,b2,b3,b4)
        allCosts(counter) = newcost;
        allAccuracies(counter) = newaccuracy;
        numberOfIterations = numberOfIterations + 1
        if newaccuracy >= 0.97
            break;
        end
    end
    
    figure(2)
    clf
    semilogy([1:1e4:maximumNumberOfIterations],allCosts(1:1e4:maximumNumberOfIterations),'b-','LineWidth',2)
    hold on;
    xlabel('Iteration Number')
    ylabel('Cost')
    set(gca,'FontWeight','Bold','FontSize',18)
    print -dpng pic_cost.png

    figure(3)
    clf
    semilogy([1:1e4:maximumNumberOfIterations],allAccuracies(1:1e4:maximumNumberOfIterations),'r-','LineWidth',2)
    xlabel('Iteration Number')
    ylabel('Accuracy')
    set(gca,'FontWeight','Bold','FontSize',18)
    print -dpng pic_accuracy.png
    
    %%%%%%%%%%% Display shaded and unshaded regions 
    N = 500;
    Dx = 1/N;
    Dy = 1/N;
    xvals = [0:Dx:1];
    yvals = [0:Dy:1];
    for k1 = 1:N+1
        xk = xvals(k1);
        for k2 = 1:N+1
            yk = yvals(k2);
            xy = [xk;yk];
            a2 = activate(xy,W2,b2);
            a3 = activate(a2,W3,b3);
            a4 = activate(a3,W4,b4);
            Aval(k2,k1) = a4(1);
            Bval(k2,k1) = a4(2);
         end
    end
    [X,Y] = meshgrid(xvals,yvals);
    
    figure(4)
    clf
    a2 = subplot(1,1,1);
    Mval = Aval>Bval;
    contourf(X,Y,Mval,[0.5 0.5])
    hold on
    colormap([1 1 1; 0.8 0.8 0.8])
    plot(x1(firstHalfOfPointsRange),x2(firstHalfOfPointsRange),'ro','MarkerSize',12,'LineWidth',4)
    plot(x1(secondHalfOfPointsRange),x2(secondHalfOfPointsRange),'bx','MarkerSize',12,'LineWidth',4)
    a2.XTick = [0 1];
    a2.YTick = [0 1];
    a2.FontWeight = 'Bold';
    a2.FontSize = 16;
    xlim([0,1])
    ylim([0,1])

    function [costval, accuracy] = cost(W2,W3,W4,b2,b3,b4)
     numberOfPointsClassifiedCorrectly = 0;
     costvec = zeros(numberOfPoints,1); 
     for i = 1:numberOfPoints
         x =[x1(i);x2(i)];
         a2 = activate(x,W2,b2);
         a3 = activate(a2,W3,b3);
         a4 = activate(a3,W4,b4);
    
         category = zeros(2, 1);
         if a4(1, 1) > a4(2, 1)
            category = [1 ; 0];
         end
         if a4(1, 1) < a4(2, 1)
            category = [0 ; 1];
         end
         
         if isequal(category, y(:, i))
            numberOfPointsClassifiedCorrectly = numberOfPointsClassifiedCorrectly + 1;
         end
    
         costvec(i, 1) = norm(y(:,i) - a4,2);
     end
     costval = norm(costvec,2)^2;
     accuracy = numberOfPointsClassifiedCorrectly / numberOfPoints;
    end
end
