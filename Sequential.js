//This is a very rough model, and I am still trying to work out a better version.

//IN context this model will use the input as an array and the output as binary classification so a 0 or 1.
//So a input/output pair might look like [3.14, 5.41, 1.14, 1.56, 1.86], 1




class LR{
    //First step is to create the layers, for example if we declare segment_size (OR input_size) as 10 and we declare each layer like [25, 10, 1] 1 being the output layer
    //[25, 10, 1] are all each a layer, so in total we have 3 layers, 1 with 25 units, 1 with 10 units, 1 with 1 unit
    //Setup layers Ex: (10, [25, 10, 1]) L1: A = [1 x 10] W = [10 x 25] b = [1 x 25], L2: A = [1 x 25] W = [25 x 10] b = [1 x 10], l3: A = [1 x 10] W = [10 x 1] b = [1 x 1] *A = input*

    //You can check if the math will work between matrices by looking at the input and W parameter of each layer.
    //For example: L1: A = [1 x 10] W = [10 x 25] this will result in a [1 x 25] as long as the 2nd number of the 1st array and the 1st number of the 2nd array match.
    //[1 x *10*] [*10* x 25] Look at the "inner" numbers and see if they match.

    //We store the parameters W and b in this.layers, we also store the activation "g" as either relu or sigmoid.
    //It will be relu for every layer besides the last layer in this context.
    constructor(segment_size, layer_sizes){
        this.n = segment_size
        this.alpha = alpha;
        this.layers = [];

        let input_dim = segment_size;
        for(let i = 0; i < layer_sizes.length; i++){
            let size = layer_sizes[i];
            let W = Array.from({ length: input_dim }, () => Array(size).fill(Math.random() * iniW));
            let b = Array(size).fill(Math.random() * iniB);
            let activation = i === layer_sizes.length - 1 ? this.sigmoid : this.relu;

            this.layers.push({ W, b, g: activation });
            input_dim = size;
        }
        
    }
    
    //Dense_layer incorperates w * x + b, which is our forward pass, but it will do this for each unit in our layer *a_in = input*
    //Each a_out will be an array as long as the layer size, so we will see a_out being an array of 25 then 10 then 1.
    
    dense_layer(a_in, W, b, g){
        let units = W[0].length;
        let a_out = Array(units).fill(0);

        for(let j = 0; j < units; j++){
            let z = a_in.reduce((sum, a, i) => sum + a * W[i][j], 0) + b[j];
            a_out[j] = g(z);
        }

        return a_out;
    }

    //Sequential will get information from each layer and pass them down into dense_layer
    //This will produce an output which is a after the for loop is over *We only use [0] to get rid of the brackets, as in this case our last layer is 1 unit*
    //We also store all activations and return them for other uses if needed.
    sequential(x){
        let a = x;
        let activations = [a];
        for(let layer of this.layers){
            let { W, b, g } = layer;
            a = this.dense_layer(a, W, b, g);
            activations.push(a);
        } 

        return {output: a[0], activations};
    }

    //These are the activation functions.
    sigmoid(z){ return 1 / (1 + Math.exp(-z)); }
    sigmoid_der(z){ return z * (1 - z); }

    relu(z) { return Math.max(0, z); }
    relu_der(z) { return z > 0 ? 1 : 0; }

    //This is to predict a output, run this.sequential and return output.
    predict(x){
        let {output} = this.sequential(x);
        return output;
    }

    //Training using gradient descent
    //To begin lets break down each aspect of this training method.
    //1st - Using basic logistic regression cost for loss calcuation
    //2nd - Using "holder" parameters in "layerGradients" which then gets applied to the real weights
    //3rd - Using L2 Regulazation for parameters.
    //4th - Using a stop-early system, this is using the "best_cost", "wait" "patience" and "min_delta" variables. This is optional, but for my case it helped reduce unnecessary lag.
    train(X, y, epochs) {
        let m = X.length; 

        let best_cost = Infinity;
        let wait = 0;
        let patience = epochs > 300 ? 50 : 10;
        let min_delta = 0.001;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let layerGradients = this.layers.map(layer => ({
                dW: layer.W.map(row => row.map(() => 0)),
                db: layer.b.map(() => 0)
            }));

            let total_cost = 0;

            for (let i = 0; i < m; i++) {

                //Get error variable, and logistic_regression loss.
                
                let { output, activations } = this.sequential(X[i]);
                output = Math.min(Math.max(output, 1e-7), 1 - 1e-7);
                let error = output - y[i];
                total_cost += -y[i] * Math.log(output) - (1 - y[i]) * Math.log(1 - output);

                //Delta, which updates each parameter
                let delta = error * this.sigmoid_der(output);

                //Run over each parameter in each layer and update them in layerGradients, which will later update the real gradients.
                for (let l = this.layers.length - 1; l >= 0; l--) {
                    let { W, b, g } = this.layers[l];
                    let dW = layerGradients[l].dW;
                    let db = layerGradients[l].db;
                    let a_prev = activations[l]; 

                    for (let j = 0; j < b.length; j++) {
                        db[j] += delta;

                        for (let k = 0; k < W.length; k++) {
                            dW[k][j] += a_prev[k] * delta;

                            dW[k][j] += (alpha / m) * W[k][j]; // ----------- L2 Regularization line
                        }
                    }

                    //Update delta for new layers.
                    if (l > 0) {
                        let activation_der = g === this.relu ? this.relu_der : this.sigmoid_der;
                        delta = W.reduce((sum, w_row, j) => {
                            return sum + delta * w_row.reduce((acc, w_ij, k) => acc + w_ij * activation_der(activations[l][j]), 0);
                        }, 0);
                    }
                }
            }

            //Apply layerGradients to real gradients.
            for (let l = 0; l < this.layers.length; l++) {
                let { W, b } = this.layers[l];
                let { dW, db } = layerGradients[l];

                for (let j = 0; j < b.length; j++) {
                    b[j] -= (this.alpha / m) * db[j];

                    for (let k = 0; k < W.length; k++) {
                        W[k][j] -= (this.alpha / m) * dW[k][j];
                    }
                }
            }

            //Log average cost
            let avg_cost = total_cost / m
            if (epoch % 100 === 0 || epoch === epochs - 1) {
                log(`Epoch: ${epoch}, Cost: ${fixN(avg_cost)}`);
            }

            //If cost isnt moving more than min_delta, stop the program.
            if(avg_cost < best_cost - min_delta){
                best_cost = avg_cost;
                wait = 0;
            } else {
                wait += 1;
                if(wait >= patience){
                    log(`Stopped: ${epoch}, Cost: ${fixN(avg_cost)}`);
                    break;
                }
            }
        }
    }

}
