package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf
import scalation.mathstat.{PlotM, *}
//import scalation.modeling.Regression
//import scalation.modeling.RidgeRegression
//import scalation.modeling.SymbolicRegression
import scalation.*
//import scalation.modeling.LassoRegression.hp
//import scalation.modeling.RidgeRegression.hp

import scala.collection.mutable.Set
import scala.math.{log, exp}
//import scala.math.{abs, exp, log, sqrt}
//import scalation.random.{Normal, PermutedVecI}

//import scala.collection.mutable.{ArrayBuffer, IndexedSeq}





object autompgAquaticToxicity:
  //define the columns for the autompg aquatic toxicity
  val autompg_col = Array ("cylinders","displacement","horsepower","weight","acceleration","model year","mpg")
  //Load the data into a matrix from the excel
  val autompg = MatrixD.load("Auto.csv")
  //seperate the x and y
  val autompg_x = autompg(?, 1 to 6)
  val autompg_y = autompg(?, 0)
  print(autompg_y)

  val autompg_mu_x = autompg_x.mean // column wise mean of x
  val autompg_mu_y = autompg_y.mean // mean of y
  val autompg_x_c = autompg_x - autompg_mu_x // centered x (column wise)
  val autompg_y_c = autompg_y - autompg_mu_y // centered y

  val _1 = VectorD.one(autompg.dim) // vector of all ones
  val autompg_oxy = _1 +^: autompg // prepend a column of all ones to xy
  val autompg_ox = _1 +^: autompg_x // prepend a column of all ones to x

  val autompg_x_fname: Array [String] = autompg_col.take (6)
  val autompg_ox_fname: Array [String] = Array ("intercept") ++ autompg_x_fname

end autompgAquaticToxicity

import autompgAquaticToxicity._

@main def autompg_Correlation (): Unit =

  banner ("Variable Names in autompg Dataset")
  println (s"xr_fname = ${stringOf (autompg_x)}")                     // raw dataset
  println (s"x_fname  = ${stringOf (autompg_col)}")                      // origin column removed
  println (s"ox_fname = ${stringOf (autompg_ox_fname)}")                     // intercept (1's) added
  banner ("Correlation Analysis: reponse y vs. column x(?, j)")
  for j <- autompg_x.indices2 do
    val x_j = autompg_x(?, j)
    val correlation = autompg_y corr x_j
    val corr2       = correlation * correlation
    println (s"correlation of y vs. x(?, $j) = $correlation \t $corr2 \t ${autompg_col(j)}")
    new Plot (x_j, autompg_y, null, s"y vs, x(?, $j)")
  end for

end autompg_Correlation

@main def autompg_NullModel (): Unit =

  banner ("NullModel model: y = b₀")
  val mod = NullModel (autompg)                                           // create a null model
  mod.trainNtest ()()                                                // train and test the model

  val autompg_yp = mod.predict (autompg_x)                                           // predict y for all x rows
  val e  = autompg_y - autompg_yp                                                    // error/residual vector
  println (s"error e = $e")

  new Plot (null, autompg_y, autompg_yp, "NullModel")
  new Plot (null, e, null, "NullModel error")

end autompg_NullModel

@main def autompg_SimpleRegression (): Unit =

  banner ("SimplerRegression model: y = b₁*x₁")
  val mod = SimpleRegression (autompg_oxy)                                   // create a SimplerRegression model
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end autompg_SimpleRegression


@main def autompg_Regression (): Unit =

  banner ("Regression model: y = b₀ + b₁*x₁ + b₂*x₂ + b₃*x₃ + b₄*x₄ + b₅*x₅ + b₆*x₆")
  val mod = Regression (autompg_oxy, autompg_ox_fname)()                             // create a Regression Model (with intercept)
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end autompg_Regression

@main def autompg_linear_regression (): Unit =

  val mod = new Regression (autompg_ox, autompg_y,autompg_ox_fname)
  mod.trainNtest ()()
  println(mod.summary ())

  banner ("autompg Regression Cross-Validation Test")
  FitM.showQofStatTable (mod.crossValidate ())

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${autompg_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Linear Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end autompg_linear_regression

@main def autompg_ridge_regression (): Unit =

  import RidgeRegression.hp
  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 1.0)
  println(s"hp2 = $hp2")
  banner("RidgeRegression")
  val mod = new RidgeRegression(autompg_x, autompg_y, autompg_x_fname, hp2)
  mod.trainNtest()() // train and test the model
  println(mod.summary())

  FitM.showQofStatTable (mod.crossValidate ())

  val optLambda2 = mod.findLambda
  println("optimal Lambda value: " + optLambda2) // lambda = 40.96
  hp.updateReturn("lambda", optLambda2._1) // applies first element of optLambda tuple to hyperparam


  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${autompg_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Ridge Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end autompg_ridge_regression

@main def autompg_lasso_regression (): Unit =
  banner ("LassoRegression for AutoMPG")

  import LassoRegression.hp

  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 0.64) // try different values
  println(s"hp2 = $hp2")


  val mod = new LassoRegression (autompg_x, autompg_y,autompg_x_fname, hp2)                           // create a Lasso regression model
  mod.trainNtest ()()                                            // train and test the model
  println (mod.summary ())                                       // parameter/coefficient statistics

  banner ("Forward Selection Test")
  val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for LassoRegression", lines = true)
  println (s"rSq = $rSq")
  println (s"best (lambda, sse) = ${mod.findLambda}")

  FitM.showQofStatTable(mod.crossValidate())

  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${autompg_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Lasso Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end autompg_lasso_regression

@main def autompg_SymbolicRegression (): Unit =

  banner("Symbolic Regression")
  //val mod = SymbolicRegression.quadratic (autompg_x, autompg_y, autompg_x_fname)              // add x^2 terms

  val mod = SymbolicRegression(autompg_x, autompg_y, autompg_x_fname,Set(1,1,1,1,1), intercept=true) // add, intercept, cross-terms and given powers
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics



  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${autompg_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Symbolic Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end autompg_SymbolicRegression

@main def autompg_TransformedRegression (): Unit =

  def f(u: Double): Double = -log(1 / u - 1) // transform

  def fi(t: Double): Double = 1 / (1 + exp(-t)) // inverse transform

  val extrem = extreme(autompg_y) // (min, max) for y
  val bounds = (0.01, 0.99)
  val yy = scaleV (extrem, bounds)(autompg_y)                       // rescale to domain of transform

  banner("Transformed Regression")

  val mod = new TranRegression (autompg_ox, yy, autompg_ox_fname, Regression.hp, f,fi)

  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  val yp2 = mod.predict(autompg_ox)
  val e2 = yy - yp2

  val yp2_ = scaleV(bounds, extrem)(yp2)

  val rnk = autompg_y.iqsort // rank order for vector y
  val ry = autompg_y.reorder(rnk) // actual - red
  val ryp2 = yp2_.reorder(rnk) // TranRegression - blue

  val ys = MatrixD(ry, ryp2)
  new PlotM(null, ys.transpose)

  new Plot(null, e2, null, "e2 vs. t")

  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${autompg_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Transformed Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end autompg_TransformedRegression
