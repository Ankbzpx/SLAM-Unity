﻿using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;

public class EKF_Visualizer : MonoBehaviour
{
    [SerializeField]
    int numOfLandmarks = 1;

    [SerializeField]
    int marginRatio = 10;

    [SerializeField]
    double screenScalar = 200;

    [SerializeField]
    GameObject[] prefab;

    [SerializeField]
    Shader lineShader;

    [SerializeField]
    GameObject settingsPanel;

    [SerializeField]
    Slider landmarkSlider;
    [SerializeField]
    Slider transNoiseSlider;
    [SerializeField]
    Slider angNoiseSlider;
    [SerializeField]
    Slider rangeNoiseSlider;
    [SerializeField]
    Slider bearingNoiseSlider;

    [SerializeField]
    TextMeshProUGUI landmarkText;
    [SerializeField]
    TextMeshProUGUI transNoiseText;
    [SerializeField]
    TextMeshProUGUI angNoiseText;
    [SerializeField]
    TextMeshProUGUI rangeNoiseText;
    [SerializeField]
    TextMeshProUGUI bearingNoiseText;


    [SerializeField]
    Button settingsButton;

    bool allowInput = true;
    double marginPixelSize, worldLimitX, worldLimitY;
    float UIScalar = 10f;

    RectTransform emptyRect;

    EKF_SLAM ekf_slam;

    RectTransform robotActualRectTransform, robotMeanRectTransform, robotCovRectTransform;
    List<RectTransform> landmarkActualRectTransformList = new List<RectTransform>();
    List<RectTransform> landmarkMeanRectTransformList = new List<RectTransform>();
    List<RectTransform> landmarkCovRectTransformList = new List<RectTransform>();

    const double CONFIDENCE = 0.95;

    const double Deg2Rad = Math.PI / 180;
    const double Rad2Deg = 180 / Math.PI;


    private void Awake()
    {
        Screen.SetResolution(2880, 1440, false);

        //Margin is defined by the screen height
        marginPixelSize = Screen.height / marginRatio;
        worldLimitX = ScreenXToWorld(Screen.width - marginPixelSize);
        worldLimitY = ScreenYToWorld(Screen.height - marginPixelSize);
        ekf_slam = GetComponent<EKF_SLAM>();
        settingsButton.onClick.AddListener(delegate { ToggleSettings(); });
    }

    void Start()
    {
        emptyRect = Instantiate(prefab[2], transform).GetComponent<RectTransform>();
        ekf_slam.InitializeLandmarks(numOfLandmarks, worldLimitX, worldLimitY);

        //Update game settings
        numOfLandmarks = (int)landmarkSlider.value;
        ekf_slam.transNoiseFactor = transNoiseSlider.value;
        ekf_slam.angNoiseFactor = angNoiseSlider.value;
        ekf_slam.rangeNoiseFactor = rangeNoiseSlider.value;
        ekf_slam.bearingNoiseFactor = bearingNoiseSlider.value;
    }

    private void Update()
    {
        //Update text
        landmarkText.text = ((int)landmarkSlider.value).ToString();
        transNoiseText.text = transNoiseSlider.value.ToString();
        angNoiseText.text = angNoiseSlider.value.ToString();
        rangeNoiseText.text = rangeNoiseSlider.value.ToString();
        bearingNoiseText.text = bearingNoiseSlider.value.ToString();


        if (Input.GetMouseButtonDown(0) && allowInput)
        {
            Vector2 mousePosition = Input.mousePosition;

            //Debug.Log("MousePosition" + mousePosition);
            if (!RectTransformUtility.RectangleContainsScreenPoint(settingsButton.GetComponent<RectTransform>(), mousePosition, Camera.main))
                ekf_slam.SetTargetWorldPoints(ScreenXToWorld(mousePosition.x), ScreenYToWorld(mousePosition.y));
                
        }
        else if (Input.touchCount >= 1 && allowInput)
        {
            Vector2 touchPosition = Input.touches[0].position;

            if (!RectTransformUtility.RectangleContainsScreenPoint(settingsButton.GetComponent<RectTransform>(), touchPosition, Camera.main))
                ekf_slam.SetTargetWorldPoints(ScreenXToWorld(touchPosition.x), ScreenYToWorld(touchPosition.y));
        }
    }

    public void VisualizerRobotRegistration(float x, float y, float theta)
    {
        //UI initialization
        PrefabInitialization(out robotActualRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, new Color(0, 0, 0, 0.7f), "RobActual");
        PrefabInitialization(out robotMeanRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, Color.black, "RobPredict");
        PrefabInitialization(out robotCovRectTransform, prefab[1], robotMeanRectTransform, Vector2.zero, false, Quaternion.identity, 1f, new Color(0, 0, 0, 0.3f), "RobCov");
    }

    //Instaniate new landmark ellipses
    public void VisualizerLandmarkRegistration(int landmarkLocalId)
    {
        RectTransform landmarkMean, landmarkCov;
        PrefabInitialization(out landmarkMean, prefab[1], transform, Vector2.zero, true, Quaternion.identity, UIScalar, UnityEngine.Random.ColorHSV(0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f), "landmark mean " + landmarkLocalId);
        landmarkMeanRectTransformList.Add(landmarkMean);
        PrefabInitialization(out landmarkCov, prefab[1], landmarkMean, Vector2.zero, false, Quaternion.identity, 1f, UnityEngine.Random.ColorHSV(0f, 1f, 1f, 1f, 0.5f, 1f, 0.7f, 0.7f), "landmark cov" + landmarkLocalId);
        landmarkCovRectTransformList.Add(landmarkCov);
    }


    public void VisualizeTrueLandmarks(int id, float x, float y)
    {
        RectTransform rect;
        PrefabInitialization(out rect, prefab[1], transform, new Vector2(x, y), true, Quaternion.identity, UIScalar, new Color(255, 0, 0, 0.7f), "True landmark " + id);
        landmarkActualRectTransformList.Add(rect);
    }

    public void Visualize(Vector<double> X_predict, Vector<double> X_actual, Matrix<double> P)
    {
        int numOfStates = X_predict.Count;

        //Visualize robots
        Vector<double> mean_r = X_predict.SubVector(0, 2);
        Matrix<double> cov_r = P.SubMatrix(0, 2, 0, 2);

        //Visualize robots
        Vector<double> mean_r_true = X_actual.SubVector(0, 2);
        Matrix<double> cov_r_true = P.SubMatrix(0, 2, 0, 2);

        //robotMeanRectTransform.rotation = Quaternion.Euler(0, 0, X[2]);

        robotMeanRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * X_predict[2])));
        robotMeanRectTransform.localPosition = new Vector2(WorldXToScreen(mean_r[0]), WorldYToScreen(mean_r[1]));

        robotActualRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * X_actual[2])));
        robotActualRectTransform.localPosition = new Vector2(WorldXToScreen(mean_r_true[0]), WorldYToScreen(mean_r_true[1]));

        CovarianceVisualization(cov_r, CONFIDENCE, robotCovRectTransform);

        if (numOfStates > 3)
        {
            for (int i = 0; i < landmarkCovRectTransformList.Count; i++)
            {
                Vector<double> mean_m = X_predict.SubVector(3 + 2 * i, 2);
                Matrix<double> cov_m = P.SubMatrix(3 + 2 * i, 2, 3 + 2 * i, 2);
                landmarkMeanRectTransformList[i].localPosition = new Vector2(WorldXToScreen(mean_m[0]), WorldYToScreen(mean_m[1]));
                CovarianceVisualization(cov_m, CONFIDENCE, landmarkCovRectTransformList[i]);
            }
        }
    }


    // Return the parameter of Ellipse for visualization
    void CovarianceVisualization(Matrix<double> covarianceMatrix, double confidence, RectTransform covRect)
    {
        var r2 = ChiSquared.InvCDF(2, confidence);
        var eigen = covarianceMatrix.Evd();
        var eigenvectors = eigen.EigenVectors;
        var eigenvalues = eigen.EigenValues.Real();

        var result = (eigenvalues * (float)r2).PointwiseSqrt();
        double width = 2 * (float)result[0];
        width = double.IsNaN(width) ? covRect.localScale.x : width;
        double height = 2 * (float)result[1];
        height = double.IsNaN(height) ? covRect.localScale.y : height;
        double angle = Rad2Deg * Math.Atan2(eigenvectors[0, 1], eigenvectors[0, 0]);

        covRect.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)angle));
        covRect.localScale = new Vector2((float)(width /1.5), (float)(height /1.5));
    }


    //Visualize the observed landmarks
    public void VisualizeMeasurement(Dictionary<int, Vector<double>> observedLandmarkList)
    {
        foreach (var landmark in observedLandmarkList)
        {
            RectTransform lmRect = landmarkCovRectTransformList[ekf_slam.worldToLocalDictionary[landmark.Key]];
            Color lineColor = lmRect.GetComponent<Image>().color;
            DrawLine(robotMeanRectTransform.position, new Vector3(lmRect.position.x, lmRect.position.y), lineColor, Time.deltaTime);

            Vector2 truePos = landmarkActualRectTransformList[landmark.Key].position;

            DrawLine(robotActualRectTransform.position, new Vector3(truePos.x, truePos.y), Color.red, Time.deltaTime);
        }
    }


    void DrawLine(Vector3 start, Vector3 end, Color color, float duration = 0.2f)
    {
        start = new Vector3(start.x, start.y, 90f);
        end = new Vector3(end.x, end.y, 90f);

        GameObject myLine = new GameObject();
        myLine.transform.position = start;
        myLine.AddComponent<LineRenderer>();
        LineRenderer lr = myLine.GetComponent<LineRenderer>();


        lr.material = new Material(lineShader);
        lr.startColor = color;
        lr.endColor = color;

        lr.startWidth = 0.2f;
        lr.endWidth = 0.2f;

        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        Destroy(myLine, duration);
    }

    void ToggleSettings()
    {
        allowInput = false;
        settingsPanel.SetActive(!settingsPanel.activeSelf);

        if(!settingsPanel.activeSelf)
            StartCoroutine(WaitForSeconds(0.5f));
    }

    public void ResetAll()
    {

        robotMeanRectTransform = null;
        robotActualRectTransform = null;
        robotCovRectTransform = null;

        foreach (Transform child in transform)
        {
            Destroy(child.gameObject);
        }

        ekf_slam.ResetAll();

        //Update game settings
        numOfLandmarks = (int)landmarkSlider.value;
        ekf_slam.transNoiseFactor = transNoiseSlider.value;
        ekf_slam.angNoiseFactor = angNoiseSlider.value;
        ekf_slam.rangeNoiseFactor = rangeNoiseSlider.value;
        ekf_slam.bearingNoiseFactor = bearingNoiseSlider.value;
        emptyRect = Instantiate(prefab[2], transform).GetComponent<RectTransform>();

        landmarkActualRectTransformList.Clear();
        landmarkMeanRectTransformList.Clear();
        landmarkCovRectTransformList.Clear();

        ekf_slam.InitializeLandmarks(numOfLandmarks, worldLimitX, worldLimitY);

        ToggleSettings();
    }

    void PrefabInitialization(out RectTransform rectTransform, GameObject prefab, Transform transform, Vector2 initialPos, bool isWorld, Quaternion initialWorldRot, float scale, Color color, string name)
    {
        //Debug.Log("Name of object: " + name);

        rectTransform = Instantiate(prefab, transform).GetComponent<RectTransform>();

        if (isWorld)
            rectTransform.localPosition = new Vector2(WorldXToScreen(initialPos.x), WorldYToScreen(initialPos.y));
        else
            rectTransform.localPosition = new Vector2(initialPos.x, initialPos.y);

        //Debug.Log(rectTransform.localPosition);

        rectTransform.localRotation = initialWorldRot;
        rectTransform.name = name;
        rectTransform.localScale = new Vector2(1 / scale, 1 / scale);
        rectTransform.GetComponent<Image>().color = color;
    }

    IEnumerator WaitForSeconds(float waitTime)
    {
        yield return new WaitForSeconds(waitTime);
        allowInput = true;
    }

    float WorldXToScreen(double world_x)
    {
        return (float)(world_x * screenScalar - Screen.width / 2 + marginPixelSize);
    }

    float WorldYToScreen(double world_y)
    {
        return (float)(world_y * screenScalar - Screen.height / 2 + marginPixelSize);
    }

    double ScreenXToWorld(double screen_x)
    {
        return (screen_x - marginPixelSize) / screenScalar;
    }

    double ScreenYToWorld(double screen_y)
    {
        return (screen_y - marginPixelSize) / screenScalar;
    }
}
