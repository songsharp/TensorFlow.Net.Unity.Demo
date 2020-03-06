using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = UnityEngine.Random;

//using TensorFlow;
using Tensorflow;
using NumSharp;

[System.Serializable]
class Prediction
{
	public float result;
}

public class BallSpawnerController : MonoBehaviour
{
	public Transform TransformGoal;
	public Transform TransformAim;
	public GameObject PrefabBall;



	private float test = 1f;

	//private TFGraph graph;
	//private TFSession session;
	private Graph graph;
	private Session session;


	void Start()
	{
		File.WriteAllText("successful_shots.csv", "");
		TextAsset graphModel = Resources.Load("frozen.pb") as TextAsset;
		//graph = new TFGraph();
		graph = new Graph();
		graph.Import(graphModel.bytes);
		//session = new TFSession(graph);
		session = new Session(graph);

		StartCoroutine(DoShoot());
	}

	private void Update()
	{
		TransformAim.LookAt(TransformGoal);
	}

	IEnumerator DoShoot()
	{
		while (true)
		{
			//			yield return new WaitUntil(() => !Input.GetButton("Jump"));
			//			yield return new WaitUntil(() => Input.GetButton("Jump"));
			var gv2 = new Vector2(
				TransformGoal.position.x,
				TransformGoal.position.z);

			var tv2 = new Vector2(
				transform.position.x, transform.position.z);

			var dir = (gv2 - tv2).normalized;
			var dist = (gv2 - tv2).magnitude;
			var arch = 0.5f;

			var closeness = Math.Min(10f, dist) / 10f;

			//float force = GetForceRandomly(dist); //使用随机力量投篮
			//float force = GetForceFromTensorFlow(dist); //使用tf获取投篮力度
			float force = GetForceFromMagicFormula(dist); //60%也不是很Magic

			var ball = Instantiate(PrefabBall, transform.position, Quaternion.identity);
			var bc = ball.GetComponent<BallController>();
			bc.Force = new Vector3(
				dir.x * arch * closeness,
				force,//* (1f / closeness) Optional: Uncomment this to experiment with artificial shot arcs!
				dir.y * arch * closeness
			);
			bc.Distance = dist;

			yield return new WaitForSeconds(0.01f);
			MoveToRandomDistance();
		}
	}

	float GetForceFromTensorFlow(float distance)
	{
		//var runner = session.GetRunner();
		//runner.AddInput(
		//	graph["shots_input"][0],
		//new float[1, 1] { { distance } }
		//);
		//runner.Fetch(graph["shots/BiasAdd"][0]);
		//float[,] recurrent_tensor = runner.Run()[0].GetValue() as float[,];

		Tensor input = graph.OperationByName("shots_input").outputs[0];
		Tensor output = graph.OperationByName("shots/BiasAdd").outputs[0];
		var recurrent_tensor = session.run(output,
			new FeedItem(input, np.array(distance / 100).reshape(1, 1)));
		var force = recurrent_tensor[0].GetSingle();
		Debug.Log(string.Format("GetForceFromTensorFlow: {0}, {1}", distance, force));
		return force;
	}

	float GetForceRandomly(float distance)
	{
		return Random.Range(0f, 1f);
	}

	[Range(0, 10)]
	public float maxVariance;
	//这应该是作者之前训练获得的两个权重参数，命中率在60%。可以加快训练。
	float GetForceFromMagicFormula(float distance)
	{
		var variance = Random.Range(1f, maxVariance);
		return (0.125f) + (0.0317f * distance * variance);
	}

	void MoveToRandomDistance()
	{
		var newPosition = new Vector3(TransformGoal.position.x + Random.Range(2.5f, 23f), transform.parent.position.y, TransformGoal.position.z);
		transform.parent.position = newPosition;
	}
}
