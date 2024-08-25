class Complex{
    constructor(){
        this.data_holder = [];
        this.segment_size = 3;
        this.segment_row = 4;
        this.segments = [];
        this.model = new Model([4, 3], [4, 3]);
    }

    normalize_data(data){
        var mean = data.reduce((sum, value) => sum + value, 0) / data.length;
        var std = Math.sqrt(data.reduce((sum, value) => sum + (value - mean) ** 2, 0) / data.length);
        return data.map(value => (value - mean) / std);
    }
    retrieve_data(){
        if(this.data_holder.length > 0){
            let lastG = engine.history.first();
            this.data_holder.unshift(lastG.bust);
        } else {
            let last50 = engine.history.toArray(50);
            for(let i = 0; i < 50; i++){
                this.data_holder.push(last50[i].bust);
            }
        }

        //this.data_holder = this.normalize_data(this.data_holder);
        
    }

    transform_data(data){
        let matrix = [];
        for(let i = 0; i < this.segment_row; i++){
            let start = i * this.segment_size;
            let end = start + this.segment_size;
            matrix.push(data.slice(start, end));
        }
        return matrix;
    }

    segment_data(){ //Note my use case involved special preparation of the data, involving grabbing the first 50 then adding 1 after.
        if(this.data_holder.length == 50){
            this.segments.push(this.transform_data(this.data_holder.slice(0, (this.segment_row * this.segment_size))));
            for(let i = 0; i < 50; i += (this.segment_row * this.segment_size)){
                this.segments.push(this.transform_data(this.data_holder.slice(i, i + (this.segment_row * this.segment_size))));
            }
            this.segments.pop();
        }
        if(this.data_holder.length % (this.segment_row * this.segment_size) == 0 && this.data_holder.length > 50){
            this.segments.unshift(this.transform_data(this.data_holder.slice(0, (this.segment_row * this.segment_size))));
        } else if(this.data_holder.length > 50){
            this.segments.shift();
            this.segments.unshift(this.transform_data(this.data_holder.slice(0, (this.segment_row * this.segment_size))));
        }
    }

    predict(){
        this.model.train(this.segments);
        log(fixN(this.model.h));
    }
}

class Matrix{
    static init_matrix(rows, cols){
        return Array.from({length: rows}, () => Array.from({length: cols}, () => Math.random() * 2 - 1))
    }
    static init_zero(rows, cols){
        return Array.from({length: rows}, () => Array.from({length: cols}, () => 0));
    }
    static transpose(a){
        return a[0].map((_, i) => a.map(row => row[i]));
    }
    static add_matrix(a, b){
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }
    static subtract_matrix(a, b){
        return a.map((row, i) => row.map((val, j) => val - b[i][j]));
    }
    static multiply_matrix(a, b){
        return a.map((row, i) => row.map((val, j) => val * b[i][j]));
    }
    static scalar_multiply(a, scalar){
        return a.map(row => row.map(val => val * scalar));
    }
    static mat_mul(a, b){
        var result = Matrix.init_zero(a.length, b[0].length);
        return result.map((row, i) => row.map((_, j) => a[i].reduce((sum, elm, k) => sum + elm * b[k][j], 0)));
    }
    static concat(a, b){
        return a.map((row, i) => row.concat(b[i]));
    }
    static sum(matrix){
        return matrix.reduce((sum, row) => sum + row.reduce((inner, val) => inner + val, 0), 0);
    }
    static apply_function(matrix, func){
        return matrix.map(row => row.map(val => func(val)));
    }
    static sigmoid(matrix){
        return Matrix.apply_function(matrix, x => 1 / (1 + Math.exp(-x)))
    }
    static sigmoid_d(matrix){
        var sigmoid = Matrix.sigmoid(matrix);
        return Matrix.apply_function(sigmoid, val => val * (1 - val));
    }
    static tanh(matrix){
        return Matrix.apply_function(matrix, x => Math.tanh(x));
    }
    static tanh_d(matrix){
        var tanh = Matrix.tanh(matrix);
        return Matrix.apply_function(tanh, val => 1 - val * val);
    }

    static mse(predicted, actual){
        let loss = 0;
        for(let i = 0; i < predicted.length; i++){
            for(let j = 0; j < predicted[0].length; j++){
                loss += (predicted[i][j] - actual[i][j]) ** 2;
            }
            
        }
        return loss / predicted.length;
    }

    static mse_gradient(predicted, actual){
        let grad = [];
        for(let i = 0; i < predicted.length; i++){
            grad[i] = [];
            for(let j = 0; j < predicted[0].length; j++){
                grad[i][j] = 2 * (predicted[i][j] - actual[i][j] / predicted.length);
            }
        }
        return grad;
    }

}



class Model{
    constructor(input_shape, hidden_shape){
        this.input_shape = input_shape;
        this.hidden_shape = hidden_shape;
        this.lR = 0.03;

        this.init_weights();

    }

    init_weights(){
        const combined_size = this.input_shape[1] + this.hidden_shape[1];
        this.wf = Matrix.init_matrix(this.hidden_shape[1], combined_size);
        this.bf = Matrix.init_zero(this.hidden_shape[0], this.hidden_shape[1]);
        this.wi = Matrix.init_matrix(this.hidden_shape[1], combined_size);
        this.bi = Matrix.init_zero(this.hidden_shape[0], this.hidden_shape[1]);
        this.wc = Matrix.init_matrix(this.hidden_shape[1], combined_size);
        this.bc = Matrix.init_zero(this.hidden_shape[0], this.hidden_shape[1]);
        this.wo = Matrix.init_matrix(this.hidden_shape[1], combined_size);
        this.bo = Matrix.init_zero(this.hidden_shape[0], this.hidden_shape[1]);

        this.c = Matrix.init_zero(this.hidden_shape[0], this.hidden_shape[1]);
        this.h = Matrix.init_zero(this.hidden_shape[0], this.hidden_shape[1]);
    }

    forward(input){
        this.combined = Matrix.concat(input, this.h);
        this.f = Matrix.sigmoid(Matrix.add_matrix(Matrix.mat_mul(this.combined, Matrix.transpose(this.wf)), this.bf));
        this.i = Matrix.sigmoid(Matrix.add_matrix(Matrix.mat_mul(this.combined, Matrix.transpose(this.wi)), this.bi));
        this.ct = Matrix.tanh(Matrix.add_matrix(Matrix.mat_mul(this.combined, Matrix.transpose(this.wc)), this.bc));
        this.c = Matrix.add_matrix(Matrix.multiply_matrix(this.f, this.c), Matrix.multiply_matrix(this.i, this.ct));
        this.o = Matrix.sigmoid(Matrix.add_matrix(Matrix.mat_mul(this.combined, Matrix.transpose(this.wo)), this.bo));
        this.h = Matrix.multiply_matrix(this.o, Matrix.tanh(this.c));
        
    }

    backward(predicted, actual){
        let loss_grad = Matrix.mse_gradient(predicted, actual);

        let d_h = Matrix.multiply_matrix(loss_grad, Matrix.tanh(this.c));
        let d_c = Matrix.multiply_matrix(loss_grad, Matrix.multiply_matrix(this.o, Matrix.tanh_d(this.c)));
        let d_o = Matrix.multiply_matrix(d_h, Matrix.tanh(this.c));
        
        d_c = Matrix.add_matrix(d_c, Matrix.multiply_matrix(this.f, d_c));

        let d_f = Matrix.multiply_matrix(d_c, this.c);
        let d_i = Matrix.multiply_matrix(d_c, this.ct);
        let d_ct = Matrix.multiply_matrix(d_c, this.i);

        d_f = Matrix.multiply_matrix(d_f, Matrix.sigmoid_d(this.f));
        d_i = Matrix.multiply_matrix(d_i, Matrix.sigmoid_d(this.i));
        d_ct = Matrix.multiply_matrix(d_ct, Matrix.sigmoid_d(this.ct));
        d_o = Matrix.multiply_matrix(d_o, Matrix.sigmoid_d(this.o));


        let d_combined = Matrix.concat(this.h, this.c);
        
        this.d_wf = Matrix.mat_mul(Matrix.transpose(d_f), d_combined);
        this.d_wi = Matrix.mat_mul(Matrix.transpose(d_i), d_combined);
        this.d_wc = Matrix.mat_mul(Matrix.transpose(d_c), d_combined);
        this.d_wo = Matrix.mat_mul(Matrix.transpose(d_o), d_combined);

        this.d_bf = d_f;
        this.d_bi = d_i;
        this.d_bc = d_ct;
        this.d_bo = d_o;

        this.wf = Matrix.subtract_matrix(this.wf, Matrix.scalar_multiply(this.d_wf, this.lR));
        this.bf = Matrix.subtract_matrix(this.bf, Matrix.scalar_multiply(this.d_bf, this.lR));
        this.wi = Matrix.subtract_matrix(this.wi, Matrix.scalar_multiply(this.d_wi, this.lR));
        this.bi = Matrix.subtract_matrix(this.bi, Matrix.scalar_multiply(this.d_bi, this.lR));
        this.wc = Matrix.subtract_matrix(this.wc, Matrix.scalar_multiply(this.d_wc, this.lR));
        this.bc = Matrix.subtract_matrix(this.bc, Matrix.scalar_multiply(this.d_bc, this.lR));
        this.wo = Matrix.subtract_matrix(this.wo, Matrix.scalar_multiply(this.d_wo, this.lR));
        this.bo = Matrix.subtract_matrix(this.bo, Matrix.scalar_multiply(this.d_bo, this.lR));
        
    }

    train(data){
        for(let i = data.length - 1; i > -1; i--){
            this.forward(data[i]);
            this.backward(this.h, data[i]);
        }
    }
}
