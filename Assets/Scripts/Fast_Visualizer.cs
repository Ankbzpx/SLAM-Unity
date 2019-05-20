using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;

public class Fast_Visualizer : MonoBehaviour
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

    Fast_SLAM fast_slam;
    RectTransform robotActualRectTransform;
    RectTransform[] particlesTransformArray;

    List<RectTransform> landmarkActualRectTransformList = new List<RectTransform>();

    const double CONFIDENCE = 0.95;

    const double Deg2Rad = Math.PI / 180;
    const double Rad2Deg = 180 / Math.PI;

    List<RectTransform>[] landmarksParticles;


    private void Awake()
    {
        Screen.SetResolution(2880, 1440, false);

        //Margin is defined by the screen height
        marginPixelSize = Screen.height / marginRatio;
        worldLimitX = ScreenXToWorld(Screen.width - marginPixelSize);
        worldLimitY = ScreenYToWorld(Screen.height - marginPixelSize);
        fast_slam = GetComponent<Fast_SLAM>();
        settingsButton.onClick.AddListener(delegate { ToggleSettings(); });
    }

    void Start()
    {
        emptyRect = Instantiate(prefab[2], transform).GetComponent<RectTransform>();
        fast_slam.InitializeLandmarks(numOfLandmarks, worldLimitX, worldLimitY);

        //Update game settings
        numOfLandmarks = (int)landmarkSlider.value;
        fast_slam.transNoiseFactor = transNoiseSlider.value;
        fast_slam.angNoiseFactor = angNoiseSlider.value;
        fast_slam.rangeNoiseFactor = rangeNoiseSlider.value;
        fast_slam.bearingNoiseFactor = bearingNoiseSlider.value;
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
                fast_slam.SetTargetWorldPoints(ScreenXToWorld(mousePosition.x), ScreenYToWorld(mousePosition.y));
                
        }
        else if (Input.touchCount >= 1 && allowInput)
        {
            Vector2 touchPosition = Input.touches[0].position;

            if (!RectTransformUtility.RectangleContainsScreenPoint(settingsButton.GetComponent<RectTransform>(), touchPosition, Camera.main))
                fast_slam.SetTargetWorldPoints(ScreenXToWorld(touchPosition.x), ScreenYToWorld(touchPosition.y));
        }
    }

    public void VisualizerRobotRegistration(int numOfParticles, float x, float y, float theta)
    {
        PrefabInitialization(out robotActualRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, new Color(0, 0, 0, 0.7f), "RobActual");

        particlesTransformArray = new RectTransform[numOfParticles];
        for (int i = 0; i < particlesTransformArray.Length; i++)
        {
            PrefabInitialization(out particlesTransformArray[i], prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), 3*UIScalar, Color.black, "RobPredict");
        }

        landmarksParticles = new List<RectTransform>[numOfParticles];
        for (int i = 0; i < landmarksParticles.Length; i++)
        {
            landmarksParticles[i] = new List<RectTransform>();
        }
    }

    public void VisualizeLandmarkRegisteration(int i, double x, double y)
    {
        RectTransform rect;
        PrefabInitialization(out rect, prefab[1], transform, new Vector2((float)x, (float)y), true, Quaternion.identity, 1.3f * UIScalar, new Color(0, 0, 255, 0.3f), "landmark for " + i);
        landmarksParticles[i].Add(rect);
    }

    public void VisualizeLandmarkParticles(List<Vector<double>>[] X_m)
    {
        for (int i = 0; i < X_m.Length; i++)
        {
            for (int j = 0; j < X_m[i].Count; j++)
            {
                landmarksParticles[i][j].localPosition = new Vector2(WorldXToScreen(X_m[i][j][0]), WorldYToScreen(X_m[i][j][1]));
            }
        }
    }


    public void VisualizeTrueLandmarks(int id, float x, float y)
    {
        RectTransform rect;
        PrefabInitialization(out rect, prefab[1], transform, new Vector2(x, y), true, Quaternion.identity, UIScalar, new Color(255, 0, 0, 0.7f), "True landmark " + id);
        landmarkActualRectTransformList.Add(rect);
    }

    public void Visualize(Vector<double>[] X_particles_predict, Vector<double> X_actual)
    {
        for (int i = 0; i < particlesTransformArray.Length; i++)
        {
            particlesTransformArray[i].localPosition = new Vector2(WorldXToScreen(X_particles_predict[i][0]), WorldYToScreen(X_particles_predict[i][1]));
            particlesTransformArray[i].localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * X_particles_predict[i][2])));
        }

        robotActualRectTransform.localPosition = new Vector2(WorldXToScreen(X_actual[0]), WorldYToScreen(X_actual[1]));
        robotActualRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * X_actual[2])));
        

        //if (numOfStates > 3)
        //{
        //    for (int i = 0; i < landmarkCovRectTransformList.Count; i++)
        //    {
        //        Vector<double> mean_m = X_particles_predict.SubVector(3 + 2 * i, 2);
        //        landmarkMeanRectTransformList[i].localPosition = new Vector2(WorldXToScreen(mean_m[0]), WorldYToScreen(mean_m[1]));
        //    }
        //}
    }

    //Visualize the observed landmarks
    public void VisualizeMeasurement(Dictionary<int, Vector<double>> observedLandmarkList)
    {
        foreach (var landmark in observedLandmarkList)
        {
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

        particlesTransformArray = null;
        robotActualRectTransform = null;

        foreach (Transform child in transform)
        {
            Destroy(child.gameObject);
        }

        fast_slam.ResetAll();

        //Update game settings
        numOfLandmarks = (int)landmarkSlider.value;
        fast_slam.transNoiseFactor = transNoiseSlider.value;
        fast_slam.angNoiseFactor = angNoiseSlider.value;
        fast_slam.rangeNoiseFactor = rangeNoiseSlider.value;
        fast_slam.bearingNoiseFactor = bearingNoiseSlider.value;
        emptyRect = Instantiate(prefab[2], transform).GetComponent<RectTransform>();

        landmarkActualRectTransformList.Clear();

        fast_slam.InitializeLandmarks(numOfLandmarks, worldLimitX, worldLimitY);

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
