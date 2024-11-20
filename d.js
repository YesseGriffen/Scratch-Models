class LR{
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

    dense_layer(a_in, W, b, g){
        let units = W[0].length;
        let a_out = Array(units).fill(0);

        for(let j = 0; j < units; j++){
            let z = a_in.reduce((sum, a, i) => sum + a * W[i][j], 0) + b[j];
            a_out[j] = g(z);
        }

        return a_out;
    }

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

    sigmoid(z){ return 1 / (1 + Math.exp(-z)); }
    sigmoid_der(z){ return z * (1 - z); }

    relu(z) { return Math.max(0, z); }
    relu_der(z) { return z > 0 ? 1 : 0; }


    predict(x){
        let {output} = this.sequential(x);
        return output;
    }

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
                
                let { output, activations } = this.sequential(X[i]);
                output = Math.min(Math.max(output, 1e-7), 1 - 1e-7);
                let error = output - y[i];
                total_cost += -y[i] * Math.log(output) - (1 - y[i]) * Math.log(1 - output);

                
                let delta = error * this.sigmoid_der(output);

                for (let l = this.layers.length - 1; l >= 0; l--) {
                    let { W, b, g } = this.layers[l];
                    let dW = layerGradients[l].dW;
                    let db = layerGradients[l].db;
                    let a_prev = activations[l]; 

                    for (let j = 0; j < b.length; j++) {
                        db[j] += delta;

                        for (let k = 0; k < W.length; k++) {
                            dW[k][j] += a_prev[k] * delta;

                            dW[k][j] += (alpha / m) * W[k][j];
                        }
                    }

                    
                    if (l > 0) {
                        let activation_der = g === this.relu ? this.relu_der : this.sigmoid_der;
                        delta = W.reduce((sum, w_row, j) => {
                            return sum + delta * w_row.reduce((acc, w_ij, k) => acc + w_ij * activation_der(activations[l][j]), 0);
                        }, 0);
                    }
                }
            }

            
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

            let avg_cost = total_cost / m
            if (epoch % 100 === 0 || epoch === epochs - 1) {
                log(`Epoch: ${epoch}, Cost: ${fixN(avg_cost)}`);
            }

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