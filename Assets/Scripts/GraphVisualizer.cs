using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

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

    GraphSLAM graphSLAM;

    RectTransform robotPredictRectTransform, robotActualRectTransform;

    List<RectTransform> landmarkMeanRectTransformList = new List<RectTransform>();
    Queue<RectTransform> robotRectTrac_predict = new Queue<RectTransform>();
    Queue<RectTransform> robotRectTrac_actual = new Queue<RectTransform>();

    bool allowInput = true;
    float marginPixelSize, worldLimitX, worldLimitY, UIScalar = 10f;

    List<RectTransform> landmarkRectTransformList = new List<RectTransform>();

    void Awake()
    {
        //Margin is defined by the screen height
        marginPixelSize = Screen.height / marginRatio;
        worldLimitX = ScreenXToWorld(Screen.width - marginPixelSize);
        worldLimitY = ScreenYToWorld(Screen.height - marginPixelSize);
        graphSLAM = GetComponent<GraphSLAM>();
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
            if (IsInputValid(mousePosition))
                graphSLAM.SetTargetWorldPoints(ScreenXToWorld(mousePosition.x), ScreenYToWorld(mousePosition.y));
        }
        else if (Input.touchCount >= 1 && allowInput)
        {
            Vector2 touchPosition = Input.touches[0].position;

            if (IsInputValid(touchPosition))
                graphSLAM.SetTargetWorldPoints(ScreenXToWorld(touchPosition.x), ScreenYToWorld(touchPosition.y));
        }
    }

    public void VisualizerRobotRegistration(float x, float y, float theta)
    {
        //UI initialization
        PrefabInitialization(out robotActualRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, new Color(0, 0, 0, 0.7f), "RobActual");
        robotRectTrac_actual.Enqueue(robotActualRectTransform);

        PrefabInitialization(out robotPredictRectTransform, prefab[0], transform, new Vector2(x, y), true, Quaternion.Euler(0, 0, theta), UIScalar, Color.black, "RobPredict");
        robotRectTrac_predict.Enqueue(robotPredictRectTransform);
    }

    //Instaniate new landmark ellipses
    public void VisualizerLandmarkRegistration(int id, float x, float y)
    {
        RectTransform landmarkMean;
        PrefabInitialization(out landmarkMean, prefab[1], transform, new Vector2(x, y), true, Quaternion.identity, UIScalar, Color.red, "landmark " + id);
        landmarkMeanRectTransformList.Add(landmarkMean);
    }

    public void Visualize(Vector<float> X_predict, Vector<float> X_actual, bool isRecord = false)
    {
        if (isRecord)
        {
            //Avoid Overflow
            if (robotRectTrac_predict.Count >= graphSLAM.historyLength)
            {
                //Dequeue
                Destroy(robotRectTrac_predict.Dequeue().gameObject);
                Destroy(robotRectTrac_actual.Dequeue().gameObject);
            }

            PrefabInitialization(out RectTransform latest_predict, prefab[0], transform, new Vector2(X_predict[0], X_predict[1]), true, Quaternion.Euler(0, 0, Mathf.Rad2Deg * X_predict[2]), UIScalar, new Color(0, 0, 0, 0.7f), "RobPredict");
            robotRectTrac_predict.Enqueue(latest_predict);
            robotPredictRectTransform = latest_predict;

            PrefabInitialization(out RectTransform latest_actual, prefab[0], transform, new Vector2(X_actual[0], X_actual[1]), true, Quaternion.Euler(0, 0, Mathf.Rad2Deg * X_actual[2]), UIScalar, new Color(0, 0, 0, 0.3f), "RobActual");
            robotRectTrac_actual.Enqueue(latest_actual);
            robotActualRectTransform = latest_actual;
        }
        else
        {
            robotPredictRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, Mathf.Rad2Deg * X_predict[2]));
            robotPredictRectTransform.localPosition = new Vector2(WorldXToScreen(X_predict[0]), WorldYToScreen(X_predict[1]));

            robotActualRectTransform.localRotation = Quaternion.Euler(new Vector3(0, 0, Mathf.Rad2Deg * X_actual[2]));
            robotActualRectTransform.localPosition = new Vector2(WorldXToScreen(X_actual[0]), WorldYToScreen(X_actual[1]));
        }
    }

    public bool OptimizedVisualization(Queue<Vector<float>> robotTrace_predict)
    {
        if (robotRectTrac_predict.Count != robotTrace_predict.Count)
            return true;

        Vector<float>[] traceArray = new Vector<float>[robotTrace_predict.Count];
        RectTransform[] rectArray = new RectTransform[robotRectTrac_predict.Count];
        robotTrace_predict.CopyTo(traceArray, 0);
        robotRectTrac_predict.CopyTo(rectArray, 0);

        for (int i = 0; i < traceArray.Length; i++)
        {
            RectTransform rect = rectArray[i];
            Vector<float> vec = traceArray[i];

            rect.localRotation = Quaternion.Euler(new Vector3(0, 0, Mathf.Rad2Deg * vec[2]));
            rect.localPosition = new Vector2(WorldXToScreen(vec[0]), WorldYToScreen(vec[1]));

            if (i == traceArray.Length - 1)
            {
                robotPredictRectTransform = rect;
            }
        }

        Debug.Log("Visualize optimization finishes.");

        return true;
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

    bool IsInputValid(Vector2 currentPos)
    {
        bool isValid = true;

        if (currentPos.x < marginPixelSize || currentPos.x > Screen.width - marginPixelSize ||
            currentPos.y < marginPixelSize || currentPos.y > Screen.height - marginPixelSize)
            isValid = false;

        return isValid;
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
}
