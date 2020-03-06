using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = UnityEngine.Random;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;


public class BallSpawnerTFNetController : MonoBehaviour
{
	public Transform TransformGoal;
	public Transform TransformAim;
	public GameObject PrefabBall;

	[Range(0, 10)]
	public float maxVariance;

	private float test = 1f;

	//private TFGraph graph;
	//private TFSession session;

	bool isTraining = false;
	// Parameters
	float learning_rate = 0.01f;
	int display_step = 50;

	NumPyRandom rng = np.random;
	public NDArray train_X;
	public NDArray train_Y;
	public NDArray test_X;
	public NDArray test_Y;
	int n_samples;
	Session sess;
	Graph graph;
	Tensor X, Y, pred;
	RefVariable W, b;
	void Initializer()
	{		
		// Start training
		loadSession();
	}
	Tensor cost;
	Operation optimizer;
	void loadSession()
	{
		//TextAsset graphModel = Resources.Load("model.pb") as TextAsset;
		//graph = new Graph();
		//graph.Import(graphModel.bytes);
		//sess = new Session(graph);
		sess = new Session();
		
		// tf Graph Input
		X = tf.placeholder(tf.float32);
		Y = tf.placeholder(tf.float32);
		//W = tf.Variable(0.0317f, dtype: tf.float32, name: "weight");
		//b = tf.Variable( -0.125f, dtype: tf.float32, name: "bias");
		W = tf.Variable(0.3f, dtype: tf.float32, name: "weight");
		b = tf.Variable(-0.1f, dtype: tf.float32, name: "bias");
		// Construct a linear model
		pred = tf.add(tf.multiply(X, W), b);

	}
	void Training()
	{
		isTraining = true;
		// Initialize the variables (i.e. assign their default value)
		sess.run(tf.global_variables_initializer());
		n_samples = train_X.shape[0];
		// Mean squared error
		cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);
		// Gradient descent
		// Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);
		int training_epochs = 200;
		
		for (int epoch = 0; epoch < training_epochs; epoch++)
		{
			foreach (var (x, y) in zip<float>(train_X, train_Y))
				sess.run(optimizer, (X, x), (Y, y));

			// Display logs per epoch step
			if ((epoch + 1) % display_step == 0)
			{
				var c = sess.run(cost, (X, train_X), (Y, train_Y));
				Debug.Log($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
			}
		}

		Debug.Log("Optimization Finished!");
		var training_cost = sess.run(cost, (X, train_X), (Y, train_Y));
		Debug.Log($"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}");
		// Testing example

		Debug.Log("Testing... (Mean square loss Comparison)");
		var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]),
			(X, test_X), (Y, test_Y));
		Debug.Log($"Testing cost={testing_cost}");
		var diff = Math.Abs((float)training_cost - (float)testing_cost);
		Debug.Log($"Absolute mean square loss difference: {diff}");

		//SaveModel(sess);
		//loadGraph();

		isTraining = false;
	}
	public void SaveModel(Session sess)
	{
		var saver = tf.train.Saver();
		var save_path = saver.save(sess, "Assets/Resources/model.ckpt");
		tf.train.write_graph(sess.graph, "Assets/Resources", "model.pb.bytes", as_text: true);

		//FreezeGraph.freeze_graph(input_graph: "Assets/Resources/model.pb.bytes",
		//				  input_saver: "",
		//				  input_binary: false,
		//				  input_checkpoint: "Assets/Resources/model.ckpt.index",
		//				  output_node_names: "input_y",
		//				  restore_op_name: "save/restore_all",
		//				  filename_tensor_name: "save/Const:0",
		//				  output_graph: "Assets/Resources/frozen_model.pb.bytes",
		//				  clear_devices: false,
		//				  initializer_nodes: "");
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

		var recurrent_tensor = sess.run(pred, (X, np.array(distance)));
		var force = recurrent_tensor[0].GetSingle();
		//Debug.Log(string.Format("GetForceFromTensorFlow: {0}, {1}", distance, force));
		return force;
	}

	private void Update()
	{
		TransformAim.LookAt(TransformGoal);
	}
	void Start()
	{
		Initializer();
		StartCoroutine(DoShoot());
	}
	IEnumerator DoShoot()
	{
		//不投篮时训练升级
		while (true)
		{
			//			yield return new WaitUntil(() => !Input.GetButton("Jump"));
			//			yield return new WaitUntil(() => Input.GetButton("Jump"));
			if (isTraining)
			{
			}
			else
			{
				var gv2 = new Vector2(
					TransformGoal.position.x,
					TransformGoal.position.z);

				var tv2 = new Vector2(
					transform.position.x, transform.position.z);

				var dir = (gv2 - tv2).normalized;
				var dist = (gv2 - tv2).magnitude;
				var arch = 0.5f;

				var closeness = Math.Min(10f, dist) / 10f;

				//前期200个靠命中，之后靠训练的数据
				float force = SuccessCountFlag > 500 ? GetForceFromTensorFlow(dist) : GetForceRandomly(dist);
				//Debug.Log(force + "    " + force2);
				var ball = Instantiate(PrefabBall, transform.position, Quaternion.identity);
				var bc = ball.GetComponent<BallController>();
				bc.Force = new Vector3(
					dir.x * arch * closeness,
					force,//* (1f / closeness) Optional: Uncomment this to experiment with artificial shot arcs!
					dir.y * arch * closeness
				);
				bc.Distance = dist;
				bc.OnSuccessAdd += Bc_OnSuccessAdd;
				yield return new WaitForSeconds(0.002f);
				MoveToRandomDistance();
			}
		}
	}
	int SuccessCountFlag;
	List<float> testXTemp = new List<float>();
	List<float> testYTemp = new List<float>();
	List<float> trainXTemp = new List<float>();
	List<float> trainYTemp = new List<float>();
	private void Bc_OnSuccessAdd(object sender, EventArgs e)
	{
		//if(test_X==null) test_X
		var ball = (BallController)sender;
		//测试数据1/3
		if (SuccessCountFlag % 3 == 0) 
		{
			testXTemp.Add(ball.Distance);
			testYTemp.Add(ball.Force.y);
		}
		else
		{
			trainXTemp.Add(ball.Distance);
			trainYTemp.Add(ball.Force.y);
		}
		if (BallController.SuccessCount == 500)
		{
			BallController.SuccessCount = 0;
			BallController.ShotCount = 0;
		}

		//每500次命中，训练一次
		if (++SuccessCountFlag % 500 == 0)
		{
			
			test_X = np.array(testXTemp.ToArray());
			test_Y = np.array(testYTemp.ToArray());
			train_X = np.array(trainXTemp.ToArray());
			train_Y = np.array(trainYTemp.ToArray());
			Training();
			testXTemp = new List<float>();
			testYTemp = new List<float>();
			trainXTemp = new List<float>();
			 trainYTemp = new List<float>();
		}
	}



	float GetForceRandomly(float distance)
	{
		return Random.Range(0f, 1f);
	}

	float GetForceFromMagicFormula(float distance)
	{
		var variance = Random.Range(1f, maxVariance);
		return (0.125f) + (0.0317f * distance * variance);
	}

	void MoveToRandomDistance()
	{
		var newPosition = new Vector3(TransformGoal.position.x + Random.Range(5f, 20f), transform.parent.position.y, TransformGoal.position.z);// + Random.Range(-5f, 5f));
		transform.parent.position = newPosition;
	}
}
