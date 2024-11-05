//This regression is used in number predicting, in this example, we have a starting dataset of 50 points of float values like 1.23, 5.41, 14.45, etc.

//Within the context of this example, a new number is generated every few seconds, and the model is going to guess the trend of the number through a outputted number between 0.0 and 1.0.

//Data will be shown after the code.


//-----------------||Setup|| -- Not important to the model

var model;



//For general use case, we will need a function that will update when new data is acquired.

function onDataEntry(){
    if(model){
        model.test()
    }
}

function startup(){
    if(!model){
        model = new Complex()
    }
}


//Simple function to turn a number like 1.2435 to 1.24.

function fixN(n){
    return parseFloat(n.toFixed(2));
}


//-----------------||Setup|| -- End



//Initial Class, to start, we have a 1d array of about 50 float values, with 1 new value being added every event.
//Note where *** is, is where the data retrieval functions are, in my use case these are set to a certain engine used, this can be anything that can acquire the data for you!

class Complex{
    constructor(){
        this.data_holder = [];                                    //To start we create data_holder to hold initial data
        this.segment_size = 5;                                    //Segment size 5 meaning every 5 data points we create a segment to use. This is to seperate our data so we have 4 in input 1 in output.
        this.segments = [];
        this.outputs = [];

        this.lr = new LR(this.segment_size - 1);                  //Creation of logisitc regression class.



        //Get initial data points                                 *** - Start

        let last50 = Array(50).fill(Math.random() * 30)
        for(let i = 0; i < 50; i++){
            this.data_holder.push(last50);
        }
                                                                  *** - End




        //Create initial segments and outputs                    //Note we add 1 segment at the start, this 1 segment is the "catch-up" segment, because our segment_size is 5, we wont get new data all the time, only every 5. but with
                                                                 //... a catch-up segment as you see later in this.segment_data(), we will push every new value into this segment, until a new one is made, then it is repeated.
                                                                 //... this catch-up segment is used in our prediction but fully made segments are used in training.
                                                                
        
        this.catchup_segment = this.data_holder.slice(1, this.segment_size);
        for(let i = 0; i < this.data_holder.length; i += this.segment_size){
            let segmentG = this.data_holder.slice(i, i + this.segment_size);
            let [output, ...segment] = segmentG;
            this.segments.push(segment);
            this.outputs.push(output);
        }

        this.lr.train(this.segments, this.normalizeArr(this.outputs), 1000);
    }




    //Retrieve data overtime                                   *** - Start

    retrieve_data(){
        let lastG = engine.last_point
        this.data_holder.unshift(lastG);
    }
                                                              *** - End



    //Retrieve segments overtime                              // We train for 20 epochs every time a new segment is made, this is optional.
    
    segment_data(){
        if(this.data_holder.length % this.segment_size == 0 && this.data_holder.length > 50){
            this.segments.unshift(this.data_holder.slice(1, this.segment_size));
            this.outputs.unshift(this.data_holder[0]);

            this.lr.train(this.segments, this.outputs, 20);

        } else if(this.data_holder.length > 50){
            this.catchup_segment.unshift(this.data_holder[0]);
            this.catchup_segment.pop();
        }
    }

    
    

    normalizeArr(Y){                                         //The Y or outputs in our case has to be between 0 and 1 as we are working with the activation function sigmoid
        let minY = Math.min(...Y);                           //This means our output will be represented as a value like 0.43 which will then need to be classified by the user.
        let maxY = Math.max(...Y);
        return Y.map(y => (y - minY) / (maxY - minY));
    }



    test(){
        this.retrieve_data();
        this.segment_data();
        log(this.lr.compute_cost(this.segments, this.normalizeArr(this.outputs)));

        log(fixN(this.lr.predict(this.segments[0])));
    }
}


//Logistic Regression class using this.segments as our X and this.outputs as our Y

//Further clarifction into the context, lets say we have data like [2.09, 1.37,	1.19, 1.52,	4.13, 19.18, 1.14, 3.56, 1.21, 11.34]
//X = [[1.37, 1.19, 1.52, 4.13], [1.14, 3.56, 1.21, 11.34]] Y = [2.09, 19.18]

//Whenever a new number is drawn lets say now we have [7.41, 2.09, 1.37, 1.19, 1.52, 4.13, 19.18, 1.14, 3.56, 1.21, 11.34]
//We will use the catchup_segment to make actual predictions outside of training, in this example catchup_segment = [7.41, 2.09, 1.37, 1.19], this would output a number between 0 and 1.

//This is everything the code above does, then we pass these values into the regression model.

class LR{
    constructor(segment_size){                                    //This model follows the equation f(x) = g(z), with w, b being parameters
        this.n = segment_size                                     //... z = dot(w * x) + b    where dot is meaning dot product between 2 arrays.
        this.w = Array(this.n).fill(Math.random() * 0.03);        //We use cost function and basic gradient descent to update the weights. Where this.alpha is our learning rate.
        this.b = Math.random() * 0.03;                            
        this.alpha = 0.0001;                                      //If you ever find the data under or overfitting, changing this.alpha or the starting parameters might help.
        
    }

    sigmoid(z){
        return 1 / (1 + Math.exp(-z));
    }

    predict(X){
        let lc = X.reduce((sum, x_i, i) => sum + this.w[i] * x_i, 0) + this.b;
        return this.sigmoid(lc);
    }

    compute_cost(X, y){
        let m = X.length;
        let total_cost = 0;

        for(let i = 0; i < m; i++){
            let prediction = this.predict(X[i]);
            total_cost += y[i] * Math.log(prediction) + (1 - y[i]) * Math.log(1 - prediction);
        }
        return -total_cost / m;
    }

    train(X, y, epochs){
        let m = X.length;

        for(let epoch = 0; epoch < epochs; epoch++){

            let sum_w = Array(this.w.length).fill(0);
            let sum_b = 0;

            for(let i = 0; i < m; i++){
                let prediction = this.predict(X[i]);
                let error = prediction - y[i];

                for(let j = 0; j < this.w.length; j++){
                    sum_w[j] += error * X[i][j];
                }

                sum_b += error;
            }

            for(let j = 0; j < this.w.length; j++){
                this.w[j] -= (this.alpha / m) * sum_w[j];
            }

            this.b -= (this.alpha / m) * sum_b;
        
            if(epoch % 100 === 0 && epochs - epoch > 2){
                log("Epoch: " + epoch + "Cost: " + fixN(this.compute_cost(X, y)));
            }

            if(epochs - epoch < 2){
                log("Epoch: " + epoch + "Cost: " + fixN(this.compute_cost(X, y)));
            }

        }
    
    }

}
