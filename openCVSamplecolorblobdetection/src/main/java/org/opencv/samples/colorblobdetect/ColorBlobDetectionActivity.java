package org.opencv.samples.colorblobdetect;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;

public class ColorBlobDetectionActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
    private static final String  TAG              = "OCVSample::Activity";

    private boolean              mIsColorSelected1 = false;
    private boolean              mIsColorSelected2 = false;
    private Mat                  mRgba;
    private Scalar               mBlobColorRgba1;
    private Scalar               mBlobColorRgba2;
    private Scalar               mBlobColorHsv1;
    private ColorBlobDetector    mDetector1;
    private Scalar               mBlobColorHsv2;
    private ColorBlobDetector    mDetector2;
    private Mat                  mSpectrum;
    private Size                 SPECTRUM_SIZE;
    private Scalar               CONTOUR_COLOR;
    private boolean               trigger=false;
    private int              threshold = 10;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.e(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public ColorBlobDetectionActivity() {
        Log.e(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.e(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.color_blob_detection_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);


    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector1 = new ColorBlobDetector();
        mDetector2 = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba1 = new Scalar(255);
        mBlobColorRgba2 = new Scalar(255);
        mBlobColorHsv1 = new Scalar(255);
        mBlobColorHsv2 = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public boolean onTouch(View v, MotionEvent event) {
        if (mIsColorSelected1 == false) {
            int cols = mRgba.cols();
            int rows = mRgba.rows();

            int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
            int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

            int x = (int) event.getX() - xOffset;
            int y = (int) event.getY() - yOffset;

            Log.e(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

            if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

            Rect touchedRect = new Rect();

            touchedRect.x = (x > 4) ? x - 4 : 0;
            touchedRect.y = (y > 4) ? y - 4 : 0;

            touchedRect.width = (x + 4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
            touchedRect.height = (y + 4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;

            Mat touchedRegionRgba = mRgba.submat(touchedRect);

            Mat touchedRegionHsv = new Mat();
            Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

            // Calculate average color of touched region
            mBlobColorHsv1 = Core.sumElems(touchedRegionHsv);
            int pointCount = touchedRect.width * touchedRect.height;
            for (int i = 0; i < mBlobColorHsv1.val.length; i++)
                mBlobColorHsv1.val[i] /= pointCount;

            mBlobColorRgba1 = converScalarHsv2Rgba(mBlobColorHsv1);

            Log.e(TAG, "Touched rgba color: (" + mBlobColorRgba1.val[0] + ", " + mBlobColorRgba1.val[1] +
                    ", " + mBlobColorRgba1.val[2] + ", " + mBlobColorRgba1.val[3] + ")");

            mDetector1.setHsvColor(mBlobColorHsv1);

            Imgproc.resize(mDetector1.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

            mIsColorSelected1 = true;

            touchedRegionRgba.release();
            touchedRegionHsv.release();
        } else if (mIsColorSelected2 == false){
            int cols = mRgba.cols();
            int rows = mRgba.rows();

            int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
            int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

            int x = (int) event.getX() - xOffset;
            int y = (int) event.getY() - yOffset;

            Log.e(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

            if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

            Rect touchedRect = new Rect();

            touchedRect.x = (x > 4) ? x - 4 : 0;
            touchedRect.y = (y > 4) ? y - 4 : 0;

            touchedRect.width = (x + 4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
            touchedRect.height = (y + 4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;

            Mat touchedRegionRgba = mRgba.submat(touchedRect);

            Mat touchedRegionHsv = new Mat();
            Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

            // Calculate average color of touched region
            mBlobColorHsv2 = Core.sumElems(touchedRegionHsv);
            int pointCount = touchedRect.width * touchedRect.height;
            for (int i = 0; i < mBlobColorHsv2.val.length; i++)
                mBlobColorHsv2.val[i] /= pointCount;

            mBlobColorRgba2 = converScalarHsv2Rgba(mBlobColorHsv2);

            Log.e(TAG, "Touched rgba color: (" + mBlobColorRgba2.val[0] + ", " + mBlobColorRgba2.val[1] +
                    ", " + mBlobColorRgba2.val[2] + ", " + mBlobColorRgba2.val[3] + ")");

            mDetector2.setHsvColor(mBlobColorHsv2);

            Imgproc.resize(mDetector2.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

            mIsColorSelected2 = true;

            touchedRegionRgba.release();
            touchedRegionHsv.release();
        }

        return false; // don't need subsequent touch events
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        mDetector1.process(mRgba);
        List<MatOfPoint> contours1 = mDetector1.getContours();

        List<List<Integer>> points1 = new ArrayList<List<Integer>>();
        for (int i = 0; i < contours1.size(); i++){
             int x1 = 0;
             int y1=0;
            int count = 0;
            List<Point> points = contours1.get(i).toList();
            for (Point p: points) {
                x1+=p.x;
                y1+=p.y;
                count ++;
            }
            List<Integer> list = new ArrayList<Integer>();
            list.add(x1/count);
            list.add(y1/count);
            list.add(count);
            points1.add(list);
    }

        Log.e("Contour 1",points1.get(0).toString());


             mDetector2.process(mRgba);
            List<MatOfPoint> contours2 = mDetector2.getContours();

        List<List<Integer>> points2 = new ArrayList<List<Integer>>();
        for (int i = 0; i < contours2.size(); i++){
            int x1 = 0;
            int y1=0;
            int count = 0;
            List<Point> points = contours2.get(i).toList();
            for (Point p: points) {
                x1+=p.x;
                y1+=p.y;
                count ++;
            }
            List<Integer> list = new ArrayList<Integer>();
            list.add(x1/count);
            list.add(y1/count);
            list.add(count);
            points1.add(list);
        }

        Log.e("Contour 2",points2.get(0).toString());
        boolean trigger = false;
        if (points2.get(0).get(2)> threshold && points1.get(0).get(2)>threshold){
            trigger = true;
            Log.e("Detected","2 points");
        }else{
            trigger = false;
            Log.e("Not Detected","2 points");
        }

            Log.e(TAG, "Contours 1 count: " + contours1.size());
        Log.e(TAG, "Contours 2 count: " + contours2.size());

          Imgproc.drawContours(mRgba, contours1, -1, CONTOUR_COLOR);
        Imgproc.drawContours(mRgba, contours2, -1, CONTOUR_COLOR);

            //Mat colorLabel = mRgba.submat(4, 68, 4, 68);
            //colorLabel.setTo(mBlobColorRgba);

           // Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
           // mSpectrum.copyTo(spectrumLabel);

        return mRgba;
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}
