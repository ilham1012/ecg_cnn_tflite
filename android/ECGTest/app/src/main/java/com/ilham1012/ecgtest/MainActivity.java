package com.ilham1012.ecgtest;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    AssetManager assetManager;
    Interpreter.Options tfliteOptions;
    Interpreter tfLite;
    String testData =  "file:///android_asset/data_test__acharya_mod__intra_ovr_train.csv";

    List<List<String>> records;

    Button btn;
    TextView tv;

    int clickIdx;

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(5);
        tfliteOptions.setUseNNAPI(true);

        String actualModelFilename = "acharya_mod__intra_ovr_train.tflite";

        btn = (Button) findViewById(R.id.btn);
        tv = (TextView) findViewById(R.id.textView);

        clickIdx = 0;

        try {
            tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename), tfliteOptions);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(testData))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    void btnClick(View v){
        tv.setText(records.get(clickIdx).toString());
        clickIdx++;
    }
}