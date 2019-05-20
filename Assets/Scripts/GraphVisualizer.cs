using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Collections;
using System;

[RequireComponent(typeof(GraphSLAM))]
public class GraphVisualizer : MonoBehaviour
{
    [SerializeField]
    int numOfLandmarks = 1;

    [SerializeField]
    int marginRatio = 10;

    [SerializeField]
    float screenScalar = 200f;

    [SerializeField]
    GameObject[] prefab;

    [SerializeField]
    Shader lineShader;

    [SerializeField]
    GameObject settingsPanel;

    [SerializeField]
    Button settingsButton;

    GraphSLAM graphSLAM;

    RectTransform robotPredictRectTransform, robotActualRectTransform;

    List<RectTransform> landmarkMeanRectTransformList = new List<RectTransform>();
    List<RectTransform> robotRectTrac_predict = new List<RectTransform>();
    List<RectTransform> robotRectTrac_actual = new List<RectTransform>();

    bool allowInput = true;
    float marginPixelSize, worldLimitX, worldLimitY, UIScalar = 10f;

    List<RectTransform> landmarkRectTransformList = new List<RectTransform>();

    const double Deg2Rad = Math.PI / 180;
    const double Rad2Deg = 180 / Math.PI;

    void Awake()
    {
        //Margin is defined by the screen height
        marginPixelSize = Screen.height / marginRatio;
        worldLimitX = ScreenXToWorld(Screen.width - marginPixelSize);
        worldLimitY = ScreenYToWorld(Screen.height - marginPixelSize);
        graphSLAM = GetComponent<GraphSLAM>();
        settingsButton.onClick.AddListener(delegate { ToggleSettings(); });
        //optimizeButton.onClick.AddListener(delegate { graphSLAM.Optimize(); });
    }

    // Start is called before the first frame update
    void Start()
    {
        graphSLAM.InitializeLandmarks(numOfLandmarks, worldLimitX, worldLimitY);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0) && allowInput)
        {
            Vector2 mousePosition = Input.mousePosition;

            //Debug.Log("MousePosition" + mousePosition);
            if (!RectTransformUtility.RectangleContainsScreenPoint(settingsButton.GetComponent<RectTransform>(), mousePosition, Camera.main))
                graphSLAM.SetTargetWorldPoints(ScreenXToWorld(mousePosition.x), ScreenYToWorld(mousePosition.y));
        }
        else if (Input.touchCount >= 1 && allowInput)
        {
            Vector2 touchPosition = Input.touches[0].position;

            if (!RectTransformUtility.RectangleContainsScreenPoint(settingsButton.GetComponent<RectTransform>(), touchPosition, Camera.main))
                graphSLAM.SetTargetWorldPoints(ScreenXToWorld(touchPosition.x), ScreenYToWorld(touchPosition.y));
        }
    }

    public void VisualizerRobotRegistration(float x, float y, float theta)
    {
        //UI initialization
        PrefabInitialization(out robotActualRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, new Color(0, 0, 0, 0.25f), "RobActual");
        robotRectTrac_actual.Add(robotActualRectTransform);

        PrefabInitialization(out robotPredictRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, new Color(0, 0, 0, 0.5f), "RobPredict");
        robotRectTrac_predict.Add(robotPredictRectTransform);
    }

    //Instaniate new landmark ellipses
    public void VisualizerLandmarkRegistration(int id, float x, float y)
    {
        RectTransform landmarkMean;
        PrefabInitialization(out landmarkMean, prefab[1], transform, new Vector2(x, y), true, Quaternion.identity, UIScalar, Color.red, "landmark " + id);
        landmarkMeanRectTransformList.Add(landmarkMean);
    }

    public void Visualize(Vector<double> X_predict, Vector<double> X_actual, bool isRecord = false)
    {
        if (isRecord)
        {
            ////Avoid Overflow
            //if (robotRectTrac_predict.Count >= graphSLAM.historyLength)
            //{
            //    //Dequeue
            //    Destroy(robotRectTrac_predict.Last().gameObject);
            //    Destroy(robotRectTrac_actual.Last().gameObject);
            //}

            PrefabInitialization(out RectTransform latest_predict, prefab[0], transform, new Vector2((float)X_predict[0], (float)X_predict[1]), true, Quaternion.Euler(0, 0, (float)(Rad2Deg * X_predict[2])), UIScalar, new Color(0, 0, 0, 0.7f), "RobPredict");
            robotRectTrac_predict.Add(latest_predict);
            robotPredictRectTransform.localScale = new Vector2(1 / (3*UIScalar), 1 / (3*UIScalar));
            robotPredictRectTransform = latest_predict;

            PrefabInitialization(out RectTransform latest_actual, prefab[0], transform, new Vector2((float)X_actual[0], (float)X_actual[1]), true, Quaternion.Euler(0, 0, (float)(Rad2Deg * X_actual[2])), UIScalar, new Color(0, 0, 0, 0.3f), "RobActual");
            robotRectTrac_actual.Add(latest_actual);
            robotActualRectTransform.localScale = new Vector2(1 / (3 * UIScalar), 1 / (3 * UIScalar));
            robotActualRectTransform = latest_actual;
        }
        else
        {
            robotPredictRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * X_predict[2])));
            robotPredictRectTransform.localPosition = new Vector2(WorldXToScreen((float)X_predict[0]), WorldYToScreen((float)X_predict[1]));

            robotActualRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * X_actual[2])));
            robotActualRectTransform.localPosition = new Vector2(WorldXToScreen((float)X_actual[0]), WorldYToScreen((float)X_actual[1]));
        }
    }

    public bool OptimizedVisualization(List<Vector<double>> robotTrace_predict)
    {
        if (robotRectTrac_predict.Count != robotTrace_predict.Count)
            return true;

        for (int i = 0; i < robotTrace_predict.Count; i++)
        {
            RectTransform rect = robotRectTrac_predict[i];
            Vector<double> vec = robotTrace_predict[i];

            rect.localRotation = Quaternion.Euler(new Vector3(0, 0, (float)(Rad2Deg * vec[2])));
            rect.localPosition = new Vector2(WorldXToScreen((float)vec[0]), WorldYToScreen((float)vec[1]));

            if (i == robotTrace_predict.Count - 1)
            {
                robotPredictRectTransform = rect;
            }
        }

        //Debug.Log("Visualize optimization finishes.");

        return true;
    }

    public void VisualizeConstraints(int[] posePair)
    {
        //Debug.Log("Draw lines");

        DrawLine(new Vector3(robotRectTrac_predict[posePair[0]].position.x, robotRectTrac_predict[posePair[0]].position.y), 
            new Vector3(robotRectTrac_predict[posePair[1]].position.x, robotRectTrac_predict[posePair[1]].position.y), Color.black, 10*Time.deltaTime);
    }

    //Visualize the observed landmarks
    public void VisualizeMeasurement(Dictionary<int, Vector<float>> observedLandmarkList)
    {
        foreach (int landmarkId in observedLandmarkList.Keys)
        {
            RectTransform lmRect = landmarkRectTransformList[landmarkId];
            Color lineColor = lmRect.GetComponent<Image>().color;
            DrawLine(robotPredictRectTransform.position, new Vector3(lmRect.position.x, lmRect.position.y), lineColor, Time.deltaTime);
            DrawLine(robotActualRectTransform.position, new Vector3(lmRect.position.x, lmRect.position.y), Color.red, Time.deltaTime);
        }
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


    float WorldXToScreen(float world_x)
    {
        return world_x * screenScalar - Screen.width / 2 + marginPixelSize;
    }

    float WorldYToScreen(float world_y)
    {
        return world_y * screenScalar - Screen.height / 2 + marginPixelSize;
    }

    float ScreenXToWorld(float screen_x)
    {
        return (screen_x - marginPixelSize) / screenScalar;
    }

    float ScreenYToWorld(float screen_y)
    {
        return (screen_y - marginPixelSize) / screenScalar;
    }

    void ToggleSettings()
    {
        allowInput = false;
        settingsPanel.SetActive(!settingsPanel.activeSelf);

        if (!settingsPanel.activeSelf)
            StartCoroutine(WaitForSeconds(0.5f));
    }
    IEnumerator WaitForSeconds(float waitTime)
    {
        yield return new WaitForSeconds(waitTime);
        allowInput = true;
    }
}
