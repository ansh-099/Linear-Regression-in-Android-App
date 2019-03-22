package com.develop.anshulsingh.myapplication;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    EditText etText;
    TextView tvText;
    Button btn;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_NAME = "file:///android_asset/optimized_frozen_linear_regression.pb";
    private static final String INPUT_NODE = "x";
    private static final String OUTPUT_NODE = "y_output";
    private static final int[] INPUT_SHAPE = {1,1}; // try {1}
    private TensorFlowInferenceInterface inferenceInterface;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etText = findViewById(R.id.etText);
        tvText = findViewById(R.id.tvText);
        btn = findViewById(R.id.btn);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(),MODEL_NAME);

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                float input = Float.parseFloat(etText.getText().toString());
                String result = performInference(input);
                tvText.setText(result);

            }
        });


    }

    private String performInference(float input){
        float[] floatArray = {input};
        inferenceInterface.fillNodeFloat(INPUT_NODE,INPUT_SHAPE,floatArray);
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});
        float[] results = {0.0f};
        inferenceInterface.readNodeFloat(OUTPUT_NODE,results);
        String finalResult = String.valueOf(results[0]);
        return finalResult;
    }
}
