package com.example.plantsclassifierandroid;

import android.app.Activity;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private TextView textView;
    private ImageView imageView;
    private Resources resources;

    private Button cameraButton;
    private Uri storageImgUri;

    private Button galleryButton;

    private Module model;
    private String[] id_to_cls;

    // Launcher for camera activity to get result
    ActivityResultLauncher<Intent> mStartForResult = registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult result) {
                if(result.getResultCode() == Activity.RESULT_OK){
                    Uri imageUri;
                    Bitmap photo;
                    Intent data = result.getData();
                    
                    if (data != null && data.getData() != null) {
                        // Gallery selection
                        imageUri = data.getData();
                    } else {
                        // Camera capture
                        imageUri = storageImgUri;
                    }

                    try {
                        InputStream img_stream = getContentResolver().openInputStream(imageUri);
                        photo = BitmapFactory.decodeStream(img_stream);
                        img_stream.close();
                    } catch (IOException e) {
                        Log.e("Demo App", "Error decoding image", e);
                        return;
                    }

                    // Get the orientation of the photo
                    int orientation = getOrientation(storageImgUri);

                    // Rotate the photo if needed
                    if (orientation != 0) {
                        Matrix matrix = new Matrix();
                        matrix.postRotate(orientation);
                        photo = Bitmap.createBitmap(photo, 0, 0, photo.getWidth(), photo.getHeight(), matrix, true);
                    }

                    // Set the photo to the ImageView
                    imageView.setImageBitmap(photo);

                    // Perform center crop
                    int dimension = Math.min(photo.getWidth(), photo.getHeight());
                    int x = (photo.getWidth() - dimension) / 2;
                    int y = (photo.getHeight() - dimension) / 2;
                    photo = Bitmap.createBitmap(photo, x, y, dimension, dimension);

                    // Resize the photo to 224x224
                    photo = Bitmap.createScaledBitmap(photo, 224, 224, true);

                    // Convert to tensor
                    Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                        photo,
                        new float[] {0.0f, 0.0f, 0.0f},
                        new float[] {1.0f, 1.0f, 1.0f});

                    // Run the model
                    IValue output = model.forward(IValue.from(inputTensor));
                    float[] preds = output.toTensor().getDataAsFloatArray();

                    // Find 5 maximum values in preds and show their indexes
                    int[] topIndices = new int[5];
                    float[] topValues = new float[5];
                    
                    for (int i = 0; i < preds.length; i++) {
                        for (int j = 0; j < 5; j++) {
                            if (preds[i] > topValues[j]) {
                                // Shift values down
                                for (int k = 4; k > j; k--) {
                                    topValues[k] = topValues[k-1];
                                    topIndices[k] = topIndices[k-1];
                                }
                                // Insert new value
                                topValues[j] = preds[i];
                                topIndices[j] = i;
                                break;
                            }
                        }
                    }
                    
                    // Log the results
                    for (int i = 0; i < 5; i++) {
                        Log.i("Top " + (i+1), "Index: " + topIndices[i] + ", Value: " + topValues[i]);
                    }

                    // Show the results
                    StringBuilder predicted_classes = new StringBuilder();
                    for (int i = 0; i < 5; i++) {
                        predicted_classes.append(id_to_cls[topIndices[i]]).append(": ").append(topValues[i]).append("\n");
                    }
                    textView.setText(predicted_classes.toString());
                }
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // Prepare label and resources
        resources = getResources();
        textView = findViewById(R.id.text);
        imageView = findViewById(R.id.imageView);
        textView.setText(resources.getString(R.string.label_text));

        // Camera and saving photo
        storageImgUri = FileProvider.getUriForFile(this,
                "com.example.plantsclassifierandroid.FileProvider",
                new File(getFilesDir(), "camera_photos.png"));
        cameraButton = findViewById(R.id.cameraBtn);
        cameraButton.setOnClickListener(view -> {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, storageImgUri);
            mStartForResult.launch(takePictureIntent);
        });

        // Gallery
        galleryButton = findViewById(R.id.galleryBtn);
        galleryButton.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            intent.setType("image/*");
            mStartForResult.launch(intent);
        });

        // Load the model
        model = loadModel();
        if (model == null) {
            finish();
            return;
        }
        // Load the id to class mapping
        id_to_cls = resources.getStringArray(R.array.yolo_id_to_cls);
    }

    private Module loadModel() {
        String modelFileName = resources.getString(R.string.model_pth);
        File modelFile = new File(getFilesDir(), modelFileName);

        // To load model it is necessary to have it in the internal storage
        // So we need to copy it from assets
        if (!modelFile.exists()) {
            try {
                copyFileFromAssets(modelFileName, modelFile);
            } catch (IOException e) {
                Log.e("Demo App", "Error copying model file from assets", e);
                return null;
            }
        }

        // Load model
        try {
            return LiteModuleLoader.load(modelFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("Demo App", "Error during loading model ", e);
            return null;
        }
    }

    private void copyFileFromAssets(String assetFileName, File outFile) throws IOException {
        try (InputStream in = getAssets().open(assetFileName);
             java.io.FileOutputStream out = new java.io.FileOutputStream(outFile)) {
            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }

    private int getOrientation(Uri storageImgUri) {
        try {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(storageImgUri));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                                                   ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return 90;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    return 180;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    return 270;
                default:
                    return 0;
            }
        } catch (IOException e) {
            Log.e("Demo App", "Error getting orientation", e);
            return 0;
        }
    }
}