using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class PercentageController : MonoBehaviour
{
	public TextMeshPro TextMeshPro;
	
	// Update is called once per frame
	void Update ()
	{
		TextMeshPro.text = String.Format("{0}/{1} {2:0.00}%", 
			BallController.SuccessCount,
			BallController.ShotCount,
			((float)BallController.SuccessCount / (float)BallController.ShotCount) * 100f);
	}
}
