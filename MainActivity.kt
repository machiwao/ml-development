package com.example.exercise5_objectdetection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.example.exercise5_objectdetection.ml.Model
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ObjectDetection"
        private const val CAMERA_PERMISSION_REQUEST = 101
        private const val CONFIDENCE_THRESHOLD = 0.3f  // Lowered from 0.5f to detect more faces
    }

    // UI components
    private lateinit var imageView: ImageView
    private lateinit var textureView: TextureView

    // Camera components
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraDevice: CameraDevice
    private lateinit var cameraCaptureSession: CameraCaptureSession
    private lateinit var backgroundHandler: Handler
    private lateinit var backgroundThread: HandlerThread
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // ML components
    private lateinit var model: Model
    private lateinit var labels: List<String>
    private lateinit var imageProcessor: ImageProcessor

    // Drawing resources
    private val paint = Paint().apply {
        strokeWidth = 4f
        textSize = 36f
    }

    private val colors = listOf(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.MAGENTA,
        Color.YELLOW, Color.DKGRAY, Color.BLACK, Color.WHITE
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)

        // Request camera permission if needed
        checkCameraPermission()

        // Initialize ML components
        setupML()

        // Setup camera preview listener
        setupTextureViewListener()

        // Get camera service
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private fun setupML() {
        try {
            // Load model labels
            labels = FileUtil.loadLabels(this, "labels.txt")

            // We won't use the imageProcessor directly since we need precise control over the input format
            // but we'll keep the reference for potential future use
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            // Initialize the model
            model = Model.newInstance(this)
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up ML components", e)
            Toast.makeText(this, "Failed to initialize ML model", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun setupTextureViewListener() {
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
                // Handle resize if needed
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                // Process frame when texture is updated
                processFrame()
            }
        }
    }

    private fun processFrame() {
        val bitmap = textureView.bitmap ?: return

        try {
            // Log frame processing to track execution
            Log.d(TAG, "Processing new camera frame")

            // The error shows a mismatch between tensor sizes (786432 bytes expected vs 196608 bytes provided)
            // This indicates we need to match the exact input format the model expects

            // Create a properly sized input tensor by manually preparing the buffer
            val inputWidth = 256
            val inputHeight = 256
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, false)

            // Allocate buffer (4 bytes per float * width * height * 3 channels)
            val byteBuffer = ByteBuffer.allocateDirect(4 * inputWidth * inputHeight * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            // Extract pixel values
            val intValues = IntArray(inputWidth * inputHeight)
            resizedBitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)

            // Convert RGB values to float and normalize
            var pixel = 0
            for (y in 0 until inputHeight) {
                for (x in 0 until inputWidth) {
                    val value = intValues[pixel++]
                    // Extract and normalize RGB values (0-255 -> 0-1)
                    byteBuffer.putFloat((value shr 16 and 0xFF) / 255.0f) // R
                    byteBuffer.putFloat((value shr 8 and 0xFF) / 255.0f)  // G
                    byteBuffer.putFloat((value and 0xFF) / 255.0f)        // B
                }
            }
            byteBuffer.rewind()

            // Create tensor buffer with shape matching model's expected input
            val inputFeature = org.tensorflow.lite.support.tensorbuffer.TensorBuffer.createFixedSize(
                intArrayOf(1, inputWidth, inputHeight, 3),
                org.tensorflow.lite.DataType.FLOAT32
            )
            inputFeature.loadBuffer(byteBuffer)

            // Run inference
            val outputs = model.process(inputFeature)

            // Get results
            val classProbabilities = outputs.outputFeature0AsTensorBuffer.floatArray
            val boundingBoxes = outputs.outputFeature1AsTensorBuffer.floatArray

            // Log detection details
            val numDetections = classProbabilities.size / labels.size
            Log.d(TAG, "Detected $numDetections potential objects")

            // Find highest confidence detection for debugging
            var maxConfidence = 0f
            var maxClassIdx = -1
            for (i in 0 until numDetections) {
                val detectionClasses = classProbabilities.sliceArray(i * labels.size until (i + 1) * labels.size)
                val (maxIdx, confidence) = detectionClasses.withIndex()
                    .maxByOrNull { it.value }
                    ?.let { it.index to it.value }
                    ?: (0 to 0f)

                if (confidence > maxConfidence) {
                    maxConfidence = confidence
                    maxClassIdx = maxIdx
                }
            }

            if (maxClassIdx >= 0) {
                Log.d(TAG, "Highest confidence detection: ${labels.getOrNull(maxClassIdx) ?: "Unknown"} at ${maxConfidence * 100}%")
            } else {
                Log.d(TAG, "No detections above threshold")
            }

            // Draw results on bitmap
            val resultBitmap = drawDetections(
                bitmap,
                classProbabilities,
                boundingBoxes,
                labels.size
            )

            // Update UI on main thread
            runOnUiThread {
                imageView.setImageBitmap(resultBitmap)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
            e.printStackTrace() // Print full stack trace for debugging
        }
    }

    private fun drawDetections(
        originalBitmap: Bitmap,
        classProbabilities: FloatArray,
        boundingBoxes: FloatArray,
        numClasses: Int
    ): Bitmap {
        // Create mutable copy of bitmap for drawing
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val imageHeight = mutableBitmap.height
        val imageWidth = mutableBitmap.width

        // Update paint properties based on image size
        paint.textSize = imageHeight / 15f
        paint.strokeWidth = imageHeight / 85f

        // Number of detections
        val numDetections = classProbabilities.size / numClasses
        Log.d(TAG, "Drawing $numDetections detections with $numClasses classes")

        var detectionsDrawn = 0

        for (i in 0 until numDetections) {
            // Get class probabilities for this detection
            val detectionClasses = classProbabilities.sliceArray(i * numClasses until (i + 1) * numClasses)

            // Find class with highest probability
            val (maxIdx, confidence) = detectionClasses.withIndex()
                .maxByOrNull { it.value }
                ?.let { it.index to it.value }
                ?: (0 to 0f)

            // Log all detection confidences for debugging
            if (confidence > 0.1f) {  // Log even low confidence detections
                val label = labels.getOrNull(maxIdx) ?: "Unknown"
                Log.d(TAG, "Detection $i: $label with confidence ${confidence * 100}%")
            }

            // Only process detections with confidence above threshold
            if (confidence > CONFIDENCE_THRESHOLD) {
                // Get bounding box coordinates
                val bbox = boundingBoxes.sliceArray(i * 4 until (i + 1) * 4)

                // Log raw bounding box values
                Log.d(TAG, "Raw bbox: [${bbox[0]}, ${bbox[1]}, ${bbox[2]}, ${bbox[3]}]")

                // Convert normalized coordinates to pixel values
                val xCenter = bbox[0] * imageWidth
                val yCenter = bbox[1] * imageHeight
                val boxWidth = bbox[2] * imageWidth
                val boxHeight = bbox[3] * imageHeight

                val left = (xCenter - boxWidth / 2).coerceIn(0f, imageWidth.toFloat())
                val top = (yCenter - boxHeight / 2).coerceIn(0f, imageHeight.toFloat())
                val right = (xCenter + boxWidth / 2).coerceIn(0f, imageWidth.toFloat())
                val bottom = (yCenter + boxHeight / 2).coerceIn(0f, imageHeight.toFloat())

                // Log computed coordinates
                Log.d(TAG, "Drawing box at: L=$left, T=$top, R=$right, B=$bottom")

                // Only draw if the box has reasonable dimensions
                if (right > left && bottom > top && boxWidth > 10 && boxHeight > 10) {
                    // Draw bounding box
                    paint.color = colors[maxIdx % colors.size]
                    paint.style = Paint.Style.STROKE
                    canvas.drawRect(RectF(left, top, right, bottom), paint)

                    // Draw label and confidence
                    paint.style = Paint.Style.FILL
                    val label = labels.getOrNull(maxIdx) ?: "Unknown"
                    val text = "$label (${(confidence * 100).toInt()}%)"

                    // Add background for text
                    val textBounds = Rect()
                    paint.getTextBounds(text, 0, text.length, textBounds)
                    val textBgRect = RectF(
                        left,
                        top - paint.textSize - 5,
                        left + textBounds.width() + 10,
                        top
                    )

                    val textBgPaint = Paint().apply {
                        color = Color.argb(160, 0, 0, 0)  // Semi-transparent black
                    }
                    canvas.drawRect(textBgRect, textBgPaint)

                    // Draw text
                    paint.color = Color.WHITE
                    canvas.drawText(text, left + 5, top - 5, paint)

                    detectionsDrawn++
                } else {
                    Log.d(TAG, "Skipping invalid box with dimensions: ${boxWidth}x${boxHeight}")
                }
            }
        }

        // Log summary
        Log.d(TAG, "Drew $detectionsDrawn detections out of $numDetections total")

        return mutableBitmap
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        if (!checkCameraPermission()) return

        try {
            // Get back-facing camera
            val cameraId = cameraManager.cameraIdList.firstOrNull {
                val characteristics = cameraManager.getCameraCharacteristics(it)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                facing == CameraCharacteristics.LENS_FACING_BACK
            } ?: cameraManager.cameraIdList[0]

            // Open camera
            cameraManager.openCamera(cameraId, cameraStateCallback, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open camera", e)
            Toast.makeText(this, "Failed to open camera", Toast.LENGTH_SHORT).show()
        }
    }

    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCaptureSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
            Log.d(TAG, "Camera disconnected")
        }

        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()
            Log.e(TAG, "Camera error: $error")
            Toast.makeText(this@MainActivity, "Camera error: $error", Toast.LENGTH_SHORT).show()
        }
    }

    private fun createCaptureSession() {
        try {
            val surfaceTexture = textureView.surfaceTexture
            surfaceTexture?.setDefaultBufferSize(textureView.width, textureView.height)

            val surface = Surface(surfaceTexture)
            val captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)

                // Optimize camera settings for face detection
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH)

                // Adjust exposure and ISO for better face visibility
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(15, 30))
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)

                // Set the best image quality
                set(CaptureRequest.JPEG_QUALITY, 100.toByte())
            }

            cameraDevice.createCaptureSession(
                listOf(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (!::cameraDevice.isInitialized) return

                        cameraCaptureSession = session
                        try {
                            session.setRepeatingRequest(
                                captureRequestBuilder.build(),
                                null,
                                backgroundHandler
                            )
                            Log.d(TAG, "Camera capture session started")
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to start camera preview", e)
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Failed to configure camera session")
                        Toast.makeText(this@MainActivity, "Camera configuration failed", Toast.LENGTH_SHORT).show()
                    }
                },
                backgroundHandler
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error creating capture session", e)
        }
    }

    private fun checkCameraPermission(): Boolean {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST
            )
            return false
        }
        return true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground")
        backgroundThread.start()
        backgroundHandler = Handler(backgroundThread.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread.quitSafely()
        try {
            backgroundThread.join()
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background thread", e)
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()

        if (textureView.isAvailable) {
            openCamera()
        }
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    private fun closeCamera() {
        if (::cameraCaptureSession.isInitialized) {
            cameraCaptureSession.close()
        }

        if (::cameraDevice.isInitialized) {
            cameraDevice.close()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
        cameraExecutor.shutdown()
    }
}