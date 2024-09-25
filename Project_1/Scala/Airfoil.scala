package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf
import scalation.mathstat.*
//import scalation.modeling.Regression
//import scalation.modeling.RidgeRegression
//import scalation.modeling.SymbolicRegression
import scalation.*
//import scalation.modeling.LassoRegression.hp
//import scalation.modeling.RidgeRegression.hp

import scala.collection.mutable.Set
import scala.math.{exp, log}



object AirfoilSelfNoise:
  //define the columns for the airfoil aquatic toxicity
  val airfoil_col = Array ("Frequency","Angle of attack","Chord length","Free-stream velocity","Suction side displacement thickness","Scaled sound pressure level")
  //Load the data into a matrix from the excel
  val airfoil = MatrixD.load("airfoil_self_noise.csv")
  //seperate the x and y
  val airfoil_x = airfoil(?, 0 to 4)
  val airfoil_y = airfoil(?, 5)

  val airfoil_mu_x = airfoil_x.mean // column wise mean of x
  val airfoil_mu_y = airfoil_y.mean // mean of y
  val airfoil_x_c = airfoil_x - airfoil_mu_x // centered x (column wise)
  val airfoil_y_c = airfoil_y - airfoil_mu_y // centered y

  val _1 = VectorD.one(airfoil.dim) // vector of all ones
  val airfoil_oxy = _1 +^: airfoil // prepend a column of all ones to xy
  val airfoil_ox = _1 +^: airfoil_x // prepend a column of all ones to x

  val airfoil_x_fname: Array [String] = airfoil_col.take (5)
  val airfoil_ox_fname: Array [String] = Array ("intercept") ++ airfoil_x_fname

end AirfoilSelfNoise

import AirfoilSelfNoise._

@main def airfoil_Correlation (): Unit =

  banner ("Variable Names in airfoil Dataset")
  println (s"xr_fname = ${stringOf (airfoil_x)}")                     // raw dataset
  println (s"x_fname  = ${stringOf (airfoil_col)}")                      // origin column removed
  println (s"ox_fname = ${stringOf (airfoil_ox_fname)}")                     // intercept (1's) added
  banner ("Correlation Analysis: reponse y vs. column x(?, j)")
  for j <- airfoil_x.indices2 do
    val x_j = airfoil_x(?, j)
    val correlation = airfoil_y corr x_j
    val corr2       = correlation * correlation
    println (s"correlation of y vs. x(?, $j) = $correlation \t $corr2 \t ${airfoil_col(j)}")
    new Plot (x_j, airfoil_y, null, s"y vs, x(?, $j)")
  end for

end airfoil_Correlation

@main def airfoil_NullModel (): Unit =

  banner ("NullModel model: y = b₀")
  val mod = NullModel (airfoil)                                           // create a null model
  mod.trainNtest ()()                                                // train and test the model

  val x  = airfoil(?, 0 to 4)                                             // predictors variables/columns
  val airfoil_yp = mod.predict (x)                                           // predict y for all x rows
  val e  = airfoil_y - airfoil_yp                                                    // error/residual vector
  println (s"error e = $e")

  new Plot (null, airfoil_y, airfoil_yp, "NullModel")
  new Plot (null, e, null, "NullModel error")

end airfoil_NullModel

@main def airfoil_SimpleRegression (): Unit =

  banner ("SimplerRegression model: y = b₁*x₁")
  val mod = SimpleRegression (airfoil_oxy)                                   // create a SimplerRegression model
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end airfoil_SimpleRegression


@main def airfoil_Regression (): Unit =

  banner ("Regression model: y = b₀ + b₁*x₁ + b₂*x₂ + b₃*x₃ + b₄*x₄ + b₅*x₅ + b₆*x₆")
  val mod = Regression (airfoil_oxy, airfoil_ox_fname)()                             // create a Regression Model (with intercept)
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end airfoil_Regression

@main def airfoil_linear_regression (): Unit =

  val mod = new Regression (airfoil_ox, airfoil_y,airfoil_ox_fname)
  mod.trainNtest ()()
  println(mod.summary ())
  banner ("airfoil Regression Cross-Validation Test")
  FitM.showQofStatTable (mod.crossValidate ())
  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${airfoil_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Linear Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end airfoil_linear_regression

@main def airfoil_ridge_regression (): Unit =

  import RidgeRegression.hp
  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 0.05) // try different values
  println(s"hp2 = $hp2")
  banner("RidgeRegression")
  val mod = new RidgeRegression(airfoil_ox, airfoil_y, airfoil_ox_fname, hp2)
  mod.trainNtest()() // train and test the model
  println(mod.summary())

  val optLambda2 = mod.findLambda
  println("optimal Lambda value: " + optLambda2) // lambda = 40.96
  hp.updateReturn("lambda", optLambda2._1) // applies first element of optLambda tuple to hyperparam

  FitM.showQofStatTable (mod.crossValidate ())
  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${airfoil_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Ridge Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end airfoil_ridge_regression

@main def airfoil_lasso_regression (): Unit =
  banner ("LassoRegression for Airfoil")

  import LassoRegression.hp

  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 0.001) // try different values
  println(s"hp2 = $hp2")

  val mod = new LassoRegression (airfoil_ox, airfoil_y,airfoil_ox_fname, hp2)                           // create a Lasso regression model
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
    println(s"k = $k, n = ${airfoil_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Lasso Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end airfoil_lasso_regression

@main def airfoil_SymbolicRegression (): Unit =

  banner("Symbolic Regression")
  //val mod = SymbolicRegression.quadratic (airfoil_x, airfoil_y, airfoil_x_fname)              // add x^2 terms

  val mod = SymbolicRegression(airfoil_x, airfoil_y, airfoil_x_fname, Set(2, 1, 0.5, 2)) // add, intercept, cross-terms and given powers
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${airfoil_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Symbolic Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end airfoil_SymbolicRegression

@main def airfoil_TransformedRegression (): Unit =

  def f(u: Double): Double = -log(1 / u - 1) // transform

  def fi(t: Double): Double = 1 / (1 + exp(-t)) // inverse transform

  val extrem = extreme(airfoil_y) // (min, max) for y
  val bounds = (0.01, 0.99)
  val yy = scaleV (extrem, bounds)(airfoil_y)                       // rescale to domain of transform

  banner("Transformed Regression")
  //val f = (sqrt _ ,  sq _,    "sqrt")

  val mod = new TranRegression (airfoil_ox, yy, airfoil_ox_fname, Regression.hp, f, fi)

  //val mod = SymbolicRegression(airfoil_x, airfoil_y, airfoil_x_fname, Set(2, 1, 0.5, 2)) // add, intercept, cross-terms and given powers
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  val yp2 = mod.predict(airfoil_ox)
  val e2 = yy - yp2

  val yp2_ = scaleV(bounds, extrem)(yp2)

  val rnk = airfoil_y.iqsort // rank order for vector y
  val ry = airfoil_y.reorder(rnk) // actual - red
  val ryp2 = yp2_.reorder(rnk) // TranRegression - blue

  val ys = MatrixD(ry, ryp2)
  new PlotM(null, ys.transpose)

  new Plot(null, e2, null, "e2 vs. t")

  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${airfoil_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Transformed Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end airfoil_TransformedRegression
