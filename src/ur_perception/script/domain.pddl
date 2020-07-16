(define (domain ur5)
  (:requirements :strips :equality)
  (:predicates

    ; Model static predicates
    (Robot ?r)
    (Obj ?o)

    ; Type static predicates
    (Conf ?r ?p)
    (Pose ?o ?p)
    (FreePos ?p)

    ; Fluents predicates
    (AtPose ?o ?p)
    (AtGrasp ?r ?o)
    (AtConf ?r ?p)
    (HandEmpty ?r)
    (CanMove ?r)
  )

  ; General movement action
  (:action moveToObj
    :parameters (?r ?o ?p1 ?p2)
    :precondition (and (Robot ?r) (Obj ?o) (AtPose ?o ?p2)
                       (AtConf ?r ?p1) (CanMove ?r) (HandEmpty ?r))
    :effect (and (AtConf ?r ?p2) (not (AtConf ?r ?p1))
                 (not (CanMove ?r)))
  )
  (:action moveToFreePos
    :parameters (?r ?p1 ?p2)
    :precondition (and (Robot ?r) (AtConf ?r ?p1)
                       (FreePos ?p2) (CanMove ?r))
    :effect (and (AtConf ?r ?p2) (not (AtConf ?r ?p1))
                 (not (CanMove ?r)))
  )

  ; Grasp movement action
  (:action grasp
    :parameters (?r ?p ?o)
    :precondition (and (Robot ?r) (Obj ?o) (AtPose ?o ?p)
                       (AtConf ?r ?p) (HandEmpty ?r))
    :effect (and (AtGrasp ?r ?o) (CanMove ?r) (FreePos ?p)
                 (not (AtPose ?o ?p)) (not (HandEmpty ?r)))
  )

  ; Place movement action
  (:action place
    :parameters (?r ?p ?o)
    :precondition (and (Robot ?r) (Obj ?o) (AtConf ?r ?p)
                       (not (HandEmpty ?r)) (AtGrasp ?r ?o))
    :effect (and (AtPose ?o ?p) (HandEmpty ?r) (CanMove ?r)
                 (not (AtGrasp ?r ?o)) (not (FreePos ?p)))
  )
)
