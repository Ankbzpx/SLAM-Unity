using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

public class MotionTestScript : MonoBehaviour
{
    #region UI_VARIABLES
    [SerializeField]
    int marginRatio = 10;

    [SerializeField]
    float screenScalar = 10f;

    [SerializeField]
    GameObject[] prefab;

    [SerializeField]
    float transVelocity = 10f;

    [SerializeField]
    float rotVelocity = 10f;

    [SerializeField]
    Shader lineShader;

    const float CONFIDENCE = 0.95f;

    bool allowInput = true;
    float marginPixelSize, worldLimitX, worldLimitY, disScreen, angScreen, ang;

    RectTransform robActualRect, robPredictRect, robCovRect;
    List<RectTransform> landmarkActualRectTransformList = new List<RectTransform>();
    List<RectTransform> landmarkPredictRectTransformList = new List<RectTransform>();
    List<RectTransform> landmarkCovRectTransformList = new List<RectTransform>();

    #endregion

    [SerializeField]
    int numOfLandmarks = 25;

    //Robot states vector
    Vector<float> X_rob_predict, X_rob_actual;

    //EKF states vector
    Vector<float> X;
    Matrix<float> P;

    static float transNoiseFactor = 0.2f, angNoiseFactor = 5f;
    Normal normal = Normal.WithMeanPrecision(0, 1);
    Matrix<float> R = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(transNoiseFactor, 2) });
    Matrix<float> Q = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(Mathf.Deg2Rad * angNoiseFactor, 2) });
    float[] measureRange = new float[] { 10f, 120, Mathf.PI / 4 };


    static int robotLandmarkIndex = 0;

    // Dictionary that stores associated landmarks
    static Dictionary<int, Vector<float>> landmarkExternalDictionary =
        new Dictionary<int, Vector<float>>();

    // Input world landmark index, output local landmark index
    static Dictionary<int, int> worldToLocalDictionary =
        new Dictionary<int, int>();


    void Start()
    {
        //Margin is defined by the screen height
        marginPixelSize = Screen.height / marginRatio;
        worldLimitX = ScreenXToWorld(Screen.width - marginPixelSize);
        worldLimitY = ScreenYToWorld(Screen.height - marginPixelSize);

        //States initialization
        X = Vector<float>.Build.DenseOfArray(new float[] { 0, 0, 0 });
        P = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { 1, 1, 1 });
        P = 0.5f * (P + P.Transpose());
        X_rob_predict = X.SubVector(0, 3);
        X_rob_actual = X_rob_predict.Clone();

        //UI initialization
        PrefabInitialization(out robActualRect, prefab[0], transform, Vector2.zero, true, Quaternion.Euler(0, 0, X_rob_predict[2]), screenScalar, new Color(0, 0, 0, 0.7f), "RobActual");
        PrefabInitialization(out robPredictRect, prefab[0], transform, Vector2.zero, true, Quaternion.Euler(0, 0, X_rob_predict[2]), screenScalar, Color.black, "RobPredict");
        PrefabInitialization(out robCovRect, prefab[1], robPredictRect, Vector2.zero, false, Quaternion.identity, 1f, new Color(0, 0, 0, 0.3f), "RobCov");

        //Initialize landmark
        for (int i = 0; i < numOfLandmarks; i++)
        {
            float x = Random.Range(0, worldLimitX);
            float y = Random.Range(0, worldLimitY);

            landmarkExternalDictionary.Add(i, Vector<float>.Build.DenseOfArray(new float[] { x, y }));

            RectTransform rect;
            PrefabInitialization(out rect, prefab[1], transform, new Vector2(x, y), true, Quaternion.identity, screenScalar, new Color(255, 0, 0, 0.7f), "True landmark " + i);
            landmarkActualRectTransformList.Add(rect);
        }
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0) && allowInput)
        {
            Vector2 mousePosition = Input.mousePosition;
            if (IsInputValid(mousePosition))
            {
                Vector<float> targetWorld = Vector<float>.Build.DenseOfArray(new float[2] { ScreenXToWorld(mousePosition.x), ScreenYToWorld(mousePosition.y) });
                CalculateMovement(out float disWorld, out ang, X_rob_actual, targetWorld);

                disScreen = disWorld * screenScalar;
                angScreen = Mathf.Abs(ang);
            }
        }
    }

    void FixedUpdate()
    {
        float ang_noise = Mathf.Pow(Mathf.Deg2Rad * angNoiseFactor, 2) * (float)normal.Sample(),
            dis_noise = Mathf.Pow(transNoiseFactor * screenScalar, 2) * (float)normal.Sample();

        if (angScreen > 0)
        {
            if (allowInput)
                allowInput = false;

            float delta_ang = Time.fixedDeltaTime * rotVelocity;
            //Noise free
            UpdateRectTransform(ref robPredictRect, 0, Mathf.Sign(ang) * delta_ang);
            StatesUpdate(ref X_rob_predict, 0, Mathf.Sign(ang) * delta_ang);
            CovarianceUpdate(ref P, 0, X_rob_predict[2]);
            
            //With noise
            UpdateRectTransform(ref robActualRect, 0, Mathf.Sign(ang) * ClampRad(delta_ang + ang_noise));
            StatesUpdate(ref X_rob_actual, 0.01f, Mathf.Sign(ang) * ClampRad(delta_ang + ang_noise));

            //Observation update
            ObserveLandmarks(out Dictionary<int, Vector<float>> observedLandmarks, X_rob_actual, dis_noise, ang_noise);
            ObservationUpdate(ref X, ref P, X_rob_actual, observedLandmarks);
            MeasurementVisualization(observedLandmarks);
            LandmarkVisualization(X.Count);
            CovarianceVisualization(P.SubMatrix(0, 2, 0, 2), CONFIDENCE, robCovRect);

            angScreen -= delta_ang;
        }
        else
        {
            if (disScreen > 0)
            {
                if (allowInput)
                    allowInput = false;

                float delta_dis = Time.fixedDeltaTime * transVelocity;

                //Noise free
                UpdateRectTransform(ref robPredictRect, delta_dis, 0);
                StatesUpdate(ref X_rob_predict, delta_dis / screenScalar, 0);
                CovarianceUpdate(ref P, delta_dis, X_rob_predict[2]);

                //With noise
                UpdateRectTransform(ref robActualRect, delta_dis + dis_noise, 0);
                StatesUpdate(ref X_rob_actual, (delta_dis + dis_noise) / screenScalar, 0);

                //Observation update
                ObserveLandmarks(out Dictionary<int, Vector<float>> observedLandmarks, X_rob_actual, dis_noise, ang_noise);
                ObservationUpdate(ref X, ref P, X_rob_actual, observedLandmarks);
                MeasurementVisualization(observedLandmarks);
                LandmarkVisualization(X.Count);
                CovarianceVisualization(P.SubMatrix(0, 2, 0, 2), CONFIDENCE, robCovRect);

                disScreen -= delta_dis;
            }
            else
            {
                if (!allowInput)
                    allowInput = true;
            }
        }
    }

    void PrefabInitialization(out RectTransform rectTransform, GameObject prefab, Transform transform, Vector2 initialPos, bool isWorld, Quaternion initialWorldRot, float scale, Color color, string name)
    {
        rectTransform = Instantiate(prefab, transform).GetComponent<RectTransform>();

        if (isWorld)
            rectTransform.localPosition = new Vector2(WorldXToScreen(initialPos.x), WorldYToScreen(initialPos.y));
        else
            rectTransform.localPosition = new Vector2(initialPos.x, initialPos.y);

        rectTransform.localRotation = initialWorldRot;
        rectTransform.name = name;
        rectTransform.localScale = new Vector2(1 / scale, 1 / scale);
        rectTransform.GetComponent<Image>().color = color;
    }

    void UpdateRectTransform(ref RectTransform rectTransform, float deltaDis, float deltaAng)
    {
        rectTransform.localPosition += deltaDis * (rectTransform.localRotation * new Vector3(1, 0, 0));
        rectTransform.Rotate(new Vector3(0, 0, 1), Mathf.Rad2Deg * deltaAng);
    }

    // Return the parameter of Ellipse for visualization
    void CovarianceVisualization(Matrix<float> cov, float confidence, RectTransform covRect)
    {
        var r2 = ChiSquared.InvCDF(2, confidence);
        var eigen = cov.Evd();
        var eigenvectors = eigen.EigenVectors;
        var eigenvalues = eigen.EigenValues.Real();

        var result = (eigenvalues * (float)r2).PointwiseSqrt();
        float width = 2 * (float)result[0];
        width = float.IsNaN(width) ? covRect.localScale.x : width;
        float height = 2 * (float)result[1];
        height = float.IsNaN(height) ? covRect.localScale.y : height;
        float angle = Mathf.Rad2Deg * Mathf.Atan2(eigenvectors[0, 1], eigenvectors[0, 0]);

        covRect.localRotation = Quaternion.Euler(new Vector3(0, 0, angle));
        covRect.localScale = new Vector2(width / 2, height / 2);
    }

    void LandmarkVisualization(int numOfStates)
    {
        if (numOfStates > 3)
        {
            for (int i = 0; i < landmarkCovRectTransformList.Count; i++)
            {
                Vector<float> mean_m = X.SubVector(3 + 2 * i, 2);
                Matrix<float> cov_m = P.SubMatrix(3 + 2 * i, 2, 3 + 2 * i, 2);
                landmarkPredictRectTransformList[i].localPosition = new Vector2(WorldXToScreen(mean_m[0]), WorldYToScreen(mean_m[1]));
                CovarianceVisualization(cov_m, CONFIDENCE, landmarkCovRectTransformList[i]);
            }
        }
    }

    //Visualize the observed landmarks
    void MeasurementVisualization(Dictionary<int, Vector<float>> observedLandmarkList)
    {
        foreach (var landmark in observedLandmarkList)
        {
            RectTransform lmRect = landmarkCovRectTransformList[worldToLocalDictionary[landmark.Key]];
            Color lineColor = lmRect.GetComponent<Image>().color;
            DrawLine(robPredictRect.position, new Vector3(lmRect.position.x, lmRect.position.y), lineColor, ref lineShader , Time.fixedDeltaTime);

            Vector2 truePos = landmarkActualRectTransformList[landmark.Key].position;

            DrawLine(robActualRect.position, new Vector3(truePos.x, truePos.y), Color.red, ref lineShader, Time.fixedDeltaTime);
        }
    }

    void StatesUpdate(ref Vector<float> currentState, float dis, float ang)
    {
        float theta_r = ClampRad(ang + currentState[2]);
        // Calculate the movement
        float dx = dis * Mathf.Cos(theta_r);
        float dy = dis * Mathf.Sin(theta_r);
        currentState = Vector<float>.Build.DenseOfArray(new float[] { currentState[0] + dx, currentState[1] + dy, theta_r });
    }

    void CovarianceUpdate(ref Matrix<float> P_est, float dis, float theta_est)
    {
        Matrix<float> J = Matrix<float>.Build.DenseOfArray(new float[3, 3] { { 1, 0, -dis * Mathf.Sin(theta_est) }, { 0, 1, dis * Mathf.Cos(theta_est) }, { 0, 0, 1 } });
        P_est.SetSubMatrix(0, 0, J * P_est.SubMatrix(0, 3, 0, 3) * J.Transpose() + Q);
    }

    void CalculateMovement(out float d, out float a, Vector<float> currentState, Vector<float> targetPos)
    {
        d = (float)(currentState.SubVector(0, 2) - targetPos).L2Norm();
        a = ClampRad(Mathf.Atan2(targetPos[1] - currentState[1], targetPos[0] - currentState[0]) - currentState[2]);
    }

    void LmJacobian(out Matrix<float> Jacobian, Vector<float> X_est, Vector<float> X_lm_est)
    {
        Vector<float> d = X_est.SubVector(0, 2) - X_lm_est;
        float r = (float)d.L2Norm();

        Jacobian = Matrix<float>.Build.DenseOfArray(new float[2, 3]
            {{(d[0])/r, (d[1])/r, 0},
            {(-d[1])/Mathf.Pow(r, 2), (d[0])/Mathf.Pow(r, 2), -1 }});
    }

    void ObserveLandmarks(out Dictionary<int, Vector<float>> observations, Vector<float> X_actual, float rangeNoise, float bearingNoise)
    {
        observations = new Dictionary<int, Vector<float>>();

        foreach (int idx in landmarkExternalDictionary.Keys)
        {
            CalculateMovement(out float r, out float b, X_actual, landmarkExternalDictionary[idx]);

            r += rangeNoise;
            b = ClampRad(b + bearingNoise);

            if (Mathf.Abs(b) <= measureRange[2] && r >= measureRange[0] && r <= measureRange[1])
            {
                observations.Add(idx, Vector<float>.Build.DenseOfArray(new float[] { r, b }));
            }
        }
    }

    void ObervationEstimate(out Vector<float> observation_est, Vector<float> X_est, Vector<float> X_lm_est)
    {
        CalculateMovement(out float r, out float b, X_est, X_lm_est);
        observation_est = Vector<float>.Build.DenseOfArray(new float[] { r, b });
    }


    // Update both robot and landmarks
    void ObservationUpdate(ref Vector<float> X_est, ref Matrix<float> P_est, Vector<float> X_rob_actual, Dictionary<int, Vector<float>> observedLandmarks)
    {
        foreach (var landmark in observedLandmarks)
        {
            int world_id = landmark.Key;
            Vector<float> z_measure = landmark.Value;

            //Observation Update
            float r_measure = z_measure[0];
            float b_measure = z_measure[1];

            //If it is a new landmark
            if (!worldToLocalDictionary.ContainsKey(world_id))
            {
                // Register the new landmark
                worldToLocalDictionary.Add(world_id, robotLandmarkIndex);
                RectTransform landmarkMean, landmarkCov;
                PrefabInitialization(out landmarkMean, prefab[1], transform, Vector2.zero, true, Quaternion.identity, screenScalar, Random.ColorHSV(0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f), "landmark mean " + robotLandmarkIndex);
                landmarkPredictRectTransformList.Add(landmarkMean);
                PrefabInitialization(out landmarkCov, prefab[1], landmarkMean, Vector2.zero, false, Quaternion.identity, 1f, Random.ColorHSV(0f, 1f, 1f, 1f, 0.5f, 1f, 0.7f, 0.7f), "landmark cov" + robotLandmarkIndex);
                landmarkCovRectTransformList.Add(landmarkCov);
                //visualizer.VisualizerLandmarkRegistration(robotLandmarkIndex);
                robotLandmarkIndex++;

                int n_states = X_est.Count;

                //Estimate landmark based on estimate states with true measurement
                float x_r = X_est[0];
                float y_r = X_est[1];
                float theta_r = X_est[2];
                float alpha = ClampRad(theta_r + b_measure);

                Vector<float> X_lm_new = Vector<float>.Build.DenseOfArray(new float[]
                {x_r + r_measure*Mathf.Cos(alpha),
                    y_r + r_measure*Mathf.Sin(alpha) });

                Vector<float> X_new = Vector<float>.Build.DenseOfArray(new float[n_states + 2]);
                X_new.SetSubVector(0, n_states, X_est);
                X_new.SetSubVector(n_states, 2, X_lm_new);

                X_est = X_new;

                Matrix<float> Jxr = Matrix<float>.Build.DenseOfArray(new float[2, 3]
                {{1, 0, -r_measure*Mathf.Sin(alpha)},
                {0, 1, r_measure*Mathf.Cos(alpha)}});

                Matrix<float> Jz = Matrix<float>.Build.DenseOfArray(new float[2, 2]
                {{Mathf.Cos(alpha), -r_measure*Mathf.Sin(alpha)},
                {Mathf.Sin(alpha), r_measure*Mathf.Cos(alpha)}});

                Matrix<float> Pvv = P_est.SubMatrix(0, 3, 0, 3);
                Matrix<float> right_top = Pvv * Jxr.Transpose();
                Matrix<float> right_buttom = Jxr * Pvv * Jxr.Transpose() + Jz * R * Jz.Transpose();

                P_est = P_est.DiagonalStack(right_buttom);

                P_est.SetSubMatrix(0, n_states, right_top);
                P_est.SetSubMatrix(n_states, 0, right_top.Transpose());

                if (n_states > 3)
                {
                    Matrix<float> Pmv = P_est.SubMatrix(3, n_states - 3, 0, 3);
                    Matrix<float> right_middle = Pmv * Jxr.Transpose();

                    P_est.SetSubMatrix(3, n_states, right_middle);
                    P_est.SetSubMatrix(n_states, 3, right_middle.Transpose());
                }

                P_est = 0.5f * (P_est + P_est.Transpose());
            }

            int local_id = worldToLocalDictionary[world_id];
            int index = 3 + local_id * 2;

            ObervationEstimate(out Vector<float> z_est, X_est, X_est.SubVector(index, 2));
            LmJacobian(out Matrix<float> h, X_est, X_est.SubVector(index, 2));

            Matrix<float> H = Matrix<float>.Build.DenseOfArray(new float[2, X_est.Count]);
            H.SetSubMatrix(0, 0, h);
            H.SetSubMatrix(0, index, -h.SubMatrix(0, 2, 0, 2));

            Matrix<float> inov = H * P_est * H.Transpose() + R;

            //Kalman gain
            Matrix<float> K = P_est * H.Transpose() * inov.Inverse();
            Vector<float> diff = z_measure - z_est;
            diff[1] = ClampRad(diff[1]);

            Vector<float> X_inc = K * diff;

            X_est += X_inc;
            X_est[2] = ClampRad(X_est[2]);

            P_est = (Matrix<float>.Build.DenseIdentity(X_est.Count, X_est.Count) - K * H) * P_est;
        }
    }

    void DrawLine(Vector3 start, Vector3 end, Color color, ref Shader shader, float duration = 0.2f)
    {
        start = new Vector3(start.x, start.y, 90f);
        end = new Vector3(end.x, end.y, 90f);

        GameObject myLine = new GameObject();
        myLine.transform.position = start;
        myLine.AddComponent<LineRenderer>();
        LineRenderer lr = myLine.GetComponent<LineRenderer>();

        lr.material = new Material(shader);
        lr.startColor = color;
        lr.endColor = color;

        lr.startWidth = 0.2f;
        lr.endWidth = 0.2f;

        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        Destroy(myLine, duration);
    }


    #region UTILITY

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

    float ClampRad(float ang)
    {
        float new_ang = ang;

        if (ang > Mathf.PI)
            new_ang -= 2 * Mathf.PI;
        else if (ang < -Mathf.PI)
            new_ang += 2 * Mathf.PI;

        return new_ang;
    }

    #endregion
}
